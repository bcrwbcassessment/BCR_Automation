"""
check_missing_students.py — BCR Compliance Check
-------------------------------------------------
Run with:  python3 check_missing_students.py

Compares the emails currently recorded in Supabase bcr_results (stored in the
"pin" column) against STUDENT_ROSTER.csv to identify students who have not
yet submitted the exam.

Output
  • Console (stdout)       — formatted table visible only to the script runner
  • MISSING_STUDENTS.txt   — same report written to BCR_Automation/ (local, private)

Privacy note
  STUDENT_ROSTER.csv contains email addresses but no student names.  This
  script's output is NEVER included in any HTML dashboard or shared artifact —
  it is a private compliance tool for the assessment coordinator only.

Credentials
  Reads Supabase credentials from .streamlit/secrets.toml (same file used by
  the Streamlit student app) or from environment variables SUPABASE_URL /
  SUPABASE_KEY if those are set.
"""

from supabase_fetch import fetch_results_df, report_missing_students

print("Fetching submitted emails from Supabase…")
df = fetch_results_df()

submitted_emails = set(df["PIN"].astype(str).unique())

report_missing_students(submitted_emails)
