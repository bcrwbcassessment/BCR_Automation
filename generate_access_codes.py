"""
generate_access_codes.py
------------------------
Reads Assessment_Report.csv, filters for students currently enrolled in
BUAD 398, calculates which assessable courses each student is missing, and
produces two output files:

  STUDENT_ROSTER.csv       — Email (lowercase), Missing_Courses          (no PII names)
  LOCAL_DEMOGRAPHICS.csv   — Email, Primary_Major, GPA_Bin,
                             Transfer_Status, Class_Standing              (no names)

Assessable courses are derived automatically from QUESTION_BANK.csv so the
two files stay in sync without manual maintenance.

Privacy design
  • No student name or Banner ID is written to any output file.
  • The email address serves as the login identifier and merge key; it is
    stored locally only and is dropped from the HTML dashboard at report time.
  • GPA is represented only as a bin (e.g. "3.5–3.9") in LOCAL_DEMOGRAPHICS.csv
    to reduce the risk of exact-match re-identification in aggregate charts.
  • Transfer_Status is derived from the Cohort column:
      "New Transfer" → "Transfer",  all others → "Native"
  • Class_Standing is taken directly from the Student Class column
    (Banner values: "Senior", "Junior", etc.)
"""

import csv
import os
import re
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ASSESSMENT_PATH      = os.path.join(BASE_DIR, "Assessment_Report.csv")
QUESTION_BANK_PATH   = os.path.join(BASE_DIR, "QUESTION_BANK.csv")
STUDENT_ROSTER_OUT   = os.path.join(BASE_DIR, "STUDENT_ROSTER.csv")
LOCAL_DEMOGRAPHICS_OUT = os.path.join(BASE_DIR, "LOCAL_DEMOGRAPHICS.csv")

TERM_CODE_RE = re.compile(r"^\d{6}$")


def normalize(text: str) -> str:
    """Strip spaces and uppercase — 'ACCT 200' → 'ACCT200'."""
    return text.replace(" ", "").upper()


def gpa_bin(raw: str) -> str:
    """
    Map a GPA string to a categorical bin.
    Bins: <2.0 | 2.0–2.4 | 2.5–2.9 | 3.0–3.4 | 3.5–3.9 | 4.0
    Returns '' if unparseable.
    """
    try:
        g = float(raw)
    except (ValueError, TypeError):
        return ""
    if g < 2.0:
        return "<2.0"
    elif g < 2.5:
        return "2.0–2.4"
    elif g < 3.0:
        return "2.5–2.9"
    elif g < 3.5:
        return "3.0–3.4"
    elif g < 4.0:
        return "3.5–3.9"
    else:
        return "4.0"


# ---------------------------------------------------------------------------
# Step 1: Derive the assessable course set from QUESTION_BANK.csv
# ---------------------------------------------------------------------------
with open(QUESTION_BANK_PATH, newline="", encoding="utf-8") as f:
    assessable_courses = sorted({row["Course"] for row in csv.DictReader(f)})

print(f"Assessable courses ({len(assessable_courses)}): {assessable_courses}")


# ---------------------------------------------------------------------------
# Step 2: Parse Assessment_Report.csv
#   Rows 0-3 are metadata; row 4 is the true header.
#   Pre-filter: keep only students where BUAD 398 == "In Progress".
# ---------------------------------------------------------------------------
students = []

with open(ASSESSMENT_PATH, newline="", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    for _ in range(4):
        next(reader)                    # skip 4 metadata rows
    headers = next(reader)              # true header row

    # Build normalised header → column-index lookup
    col = {normalize(h): i for i, h in enumerate(headers)}

    # Map each assessable course to its column index in the report
    course_col: dict[str, int] = {}
    for course in assessable_courses:
        if course in col:
            course_col[course] = col[course]
        else:
            print(f"  WARNING: '{course}' not found in Assessment_Report headers — skipped.")

    buad398_idx       = col[normalize("BUAD 398")]
    gpa_idx           = col[normalize("Overall GPA")]
    major1_idx        = col[normalize("Primary Major 1")]
    major2_idx        = col.get(normalize("Primary Major 2"))   # None if column absent
    major3_idx        = col.get(normalize("Primary Major 3"))
    cohort_idx        = col[normalize("Cohort")]
    student_class_idx = col[normalize("Student Class")]

    total_rows = filtered_rows = 0
    for row in reader:
        if not any(row):
            continue
        total_rows += 1

        # Keep only students currently enrolled in BUAD 398
        if row[buad398_idx].strip() != "In Progress":
            filtered_rows += 1
            continue

        # Identify missing assessable courses for this student
        missing: list[str] = []
        for course, idx in sorted(course_col.items()):
            value = row[idx].strip() if idx < len(row) else ""
            if not TERM_CODE_RE.match(value):
                missing.append(course)

        # Collect up to three majors; blank strings for absent values
        def safe_get(idx):
            return row[idx].strip() if (idx is not None and idx < len(row)) else ""

        raw_gpa = safe_get(gpa_idx)

        # Transfer_Status: Banner's Cohort column reliably marks cohort type.
        cohort_val      = safe_get(cohort_idx)
        transfer_status = "Transfer" if cohort_val == "New Transfer" else "Native"

        # Class_Standing: Banner's Student Class column — "Senior", "Junior", etc.
        class_standing  = safe_get(student_class_idx)

        # Email: normalize to lowercase and strip whitespace
        email = row[col[normalize("XU Email Address")]].strip().lower()

        students.append({
            "email":           email,
            "missing_courses": " ".join(missing) if missing else "None",
            "gpa_bin":         gpa_bin(raw_gpa),
            "primary_major":   safe_get(major1_idx),
            "transfer_status": transfer_status,
            "class_standing":  class_standing,
        })

print(f"\nAssessment report  : {total_rows} total rows")
print(f"Filtered out       : {filtered_rows} (BUAD 398 ≠ 'In Progress')")
print(f"Students to process: {len(students)}")


# ---------------------------------------------------------------------------
# Step 3: Write STUDENT_ROSTER.csv  — login identifier + missing courses
# ---------------------------------------------------------------------------
with open(STUDENT_ROSTER_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Email", "Missing_Courses"])
    writer.writeheader()
    for s in students:
        writer.writerow({"Email": s["email"], "Missing_Courses": s["missing_courses"]})

print(f"\nWrote {len(students)} rows → STUDENT_ROSTER.csv")


# ---------------------------------------------------------------------------
# Step 4: Write LOCAL_DEMOGRAPHICS.csv  — Email + demographics, NO names
#
# Columns intentionally excluded: name, Banner ID, Term, Gender, Citizenship,
# Ethnicity flags, Transfer Hours, Earned Hours, raw GPA, and anything that
# could narrow identification to a single student.
#
# Email is included here as the merge key so build_interactive_report.py can
# join demographics onto Supabase results; it is dropped from the HTML output
# immediately after the merge.
# ---------------------------------------------------------------------------
with open(LOCAL_DEMOGRAPHICS_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "Email",
            "Primary_Major",
            "GPA_Bin",
            "Transfer_Status",
            "Class_Standing",
        ],
    )
    writer.writeheader()
    for s in students:
        writer.writerow({
            "Email":           s["email"],
            "Primary_Major":   s["primary_major"],
            "GPA_Bin":         s["gpa_bin"],
            "Transfer_Status": s["transfer_status"],
            "Class_Standing":  s["class_standing"],
        })

print(f"Wrote {len(students)} rows → LOCAL_DEMOGRAPHICS.csv")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n=== Summary ===")
print(f"  Students in roster: {len(students)}")
print(f"  Unique email check : {len({s['email'] for s in students}) == len(students)}")

missing_counts = Counter(s["missing_courses"] for s in students)
none_count     = missing_counts.pop("None", 0)
print(f"  Students missing nothing (all courses complete) : {none_count}")
print(f"  Students with ≥1 missing course                 : {len(students) - none_count}")
print(f"  Distinct missing-course combinations            : {len(missing_counts) + (1 if none_count else 0)}")

print(f"\n=== LOCAL_DEMOGRAPHICS.csv breakdown ===")
gpa_bin_counts = Counter(s["gpa_bin"] for s in students)
print("  GPA bins:")
for bin_label in ["<2.0", "2.0–2.4", "2.5–2.9", "3.0–3.4", "3.5–3.9", "4.0"]:
    print(f"    {bin_label:<10} {gpa_bin_counts.get(bin_label, 0):>4} students")

major_counts = Counter(s["primary_major"] for s in students)
print("  Primary majors:")
for major, count in sorted(major_counts.items(), key=lambda x: -x[1]):
    print(f"    {major:<45} {count:>4} students")

xfer_counts = Counter(s["transfer_status"] for s in students)
print("  Transfer status:")
for status, count in sorted(xfer_counts.items()):
    print(f"    {status:<10} {count:>4} students")

class_counts = Counter(s["class_standing"] for s in students)
print("  Class standing:")
for standing, count in sorted(class_counts.items()):
    print(f"    {standing:<10} {count:>4} students")
