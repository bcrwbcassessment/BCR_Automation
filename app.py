"""
app.py — Business Core Reflection Assessment
---------------------------------------------
Run with:  streamlit run app.py
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from supabase import create_client

# ---------------------------------------------------------------------------
# Paths (all relative to this file so the app works from any working dir)
# ---------------------------------------------------------------------------
BASE_DIR             = Path(__file__).parent
STUDENT_ROSTER_PATH  = BASE_DIR / "STUDENT_ROSTER.csv"
QUESTION_BANK_PATH   = BASE_DIR / "QUESTION_BANK.csv"
RESULTS_PATH         = BASE_DIR / "RESULTS.csv"
IMAGES_DIR           = BASE_DIR / "images"

MAX_COURSES    = 7              # Tier 1 cap: at most this many courses per student
OPTION_LETTERS = ["A", "B", "C", "D", "E"]


# ---------------------------------------------------------------------------
# Supabase client  (one connection per process; secrets from .streamlit/secrets.toml)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)


# ---------------------------------------------------------------------------
# Data helpers  (cached so CSVs are only read once per Streamlit process)
# ---------------------------------------------------------------------------
@st.cache_data
def load_student_roster() -> dict[str, str]:
    """Returns {email: Missing_Courses_string}.
    Keys are normalized to lowercase and stripped so a typo or
    mixed-case entry never locks a student out.
    """
    roster = {}
    with open(STUDENT_ROSTER_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            roster[row["Email"].strip().lower()] = row["Missing_Courses"].strip()
    return roster


@st.cache_data
def load_question_bank() -> list[dict]:
    with open(QUESTION_BANK_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_missing_courses(missing_str: str) -> set[str]:
    """
    Convert the Missing_Courses string from STUDENT_ROSTER.csv into a normalised
    set of course codes.  "ACCT 200 FINC 300" → {'ACCT200', 'FINC300'}
    'None' (or blank) → empty set (student completed everything).
    """
    if not missing_str or missing_str.strip().lower() == "none":
        return set()
    # Normalise each token: drop spaces, uppercase
    return {t.replace(" ", "").upper() for t in missing_str.split()}


def escape_md(text: str) -> str:
    """
    Escape dollar signs so Streamlit doesn't interpret them as LaTeX delimiters
    and render the text green / in math font.  e.g. '$400' becomes '\\$400'.
    """
    return text.replace("$", "\\$")


def format_course_label(code: str) -> str:
    """Insert a space between the alpha prefix and numeric suffix.
    'ACCT200' → 'ACCT 200',  'BLAW300' → 'BLAW 300'."""
    import re
    m = re.match(r"([A-Z]+)(\d+)$", code)
    return f"{m.group(1)} {m.group(2)}" if m else code


def save_answer(question_id: str, course: str, slo: str, correct_answer: str) -> None:
    """
    on_change callback attached to every st.radio widget.

    Streamlit fires this BEFORE the rerun that follows a selection change.
    At that moment the widget's new value is already committed to session_state
    under its key, so we can read it and copy it into master_answers.

    When the student clicks Next/Previous, Streamlit destroys the off-screen
    radio widgets and removes their keys from active widget state.  master_answers
    is a plain dict in session_state — it is never touched by Streamlit's widget
    lifecycle — so every answer written here survives navigation intact.
    """
    raw      = st.session_state.get(question_id, "")
    s_letter = raw[0] if raw else ""
    st.session_state.master_answers[question_id] = {
        "PIN":            st.session_state.pin,
        "Course":         course,
        "SLO":            slo,
        "Question_ID":    question_id,
        "Student_Answer": s_letter,
        "Is_Correct":     bool(s_letter) and (s_letter == correct_answer),
    }


def generate_exam(missing_courses: set[str], all_questions: list[dict]) -> list[dict]:
    """
    Build a personalised exam using two-tier stratified randomization that
    mirrors the legacy Qualtrics pedagogy:

    Tier 1 — Course Selection (max MAX_COURSES):
        From the courses the student is eligible for (i.e. has completed —
        every course NOT listed in missing_courses), randomly sample at
        most MAX_COURSES distinct courses.  If the student is eligible
        for fewer, keep all of them.  No course is ever duplicated.

    Tier 2 — Learning Objective Stratification:
        Restrict the bank to the Tier-1 courses, then group by
        (Course, Learning Objective) and draw exactly ONE question per
        group.  This guarantees every learning objective in every
        selected course is represented exactly once.

    Final assembly:
        Concatenate the per-LO samples and shuffle the full result so
        questions from different courses are interleaved.
    """
    df = pd.DataFrame(all_questions)

    # The Learning Objective column in QUESTION_BANK.csv is "SLO" (values
    # like "LO1", "LO2", …).  If the schema ever changes, halt loudly so
    # the operator can clarify rather than silently producing a malformed
    # exam.
    LO_COLUMN = "SLO"
    if LO_COLUMN not in df.columns:
        raise KeyError(
            f"Learning Objective column '{LO_COLUMN}' not found in "
            f"QUESTION_BANK.csv. Available columns: {list(df.columns)}. "
            f"Please clarify which column represents the Learning Objective."
        )

    # Restrict to courses the student has completed
    df_eligible = df[~df["Course"].isin(missing_courses)]
    if df_eligible.empty:
        return []

    # ── Tier 1: pick up to MAX_COURSES distinct courses ───────────────
    eligible_courses = df_eligible["Course"].unique().tolist()
    n_courses        = min(MAX_COURSES, len(eligible_courses))
    selected_courses = (
        pd.Series(eligible_courses).sample(n=n_courses).tolist()
    )

    # ── Tier 2: exactly one question per (Course, LO) pair ────────────
    df_selected = df_eligible[df_eligible["Course"].isin(selected_courses)]
    per_lo_samples = [
        group.sample(n=1)
        for _, group in df_selected.groupby(["Course", LO_COLUMN])
    ]
    sampled = pd.concat(per_lo_samples, ignore_index=True)

    # ── Final shuffle: mix questions across courses ───────────────────
    sampled = sampled.sample(frac=1).reset_index(drop=True)

    return sampled.to_dict("records")


def push_to_supabase(correct: int, total: int) -> str | None:
    """
    Push a single summary row to the Supabase bcr_results table.

    Columns
    -------
    pin        text   — student access code
    major      text   — "See Master Roster" placeholder (demographics merged at report time)
    score      float  — percentage correct, rounded to 1 decimal
    timestamp  text   — ISO-8601 datetime string
    responses  text   — JSON array of every answer record from master_answers

    Returns None on success, or an error string on failure.
    """
    submission_data = None
    try:
        submission_data = {
            "pin":       str(st.session_state.pin),
            "major":     "See Master Roster",
            "score":     round(correct / total * 100, 1) if total else 0.0,
            "timestamp": datetime.now().isoformat(),
            "responses": json.dumps(list(st.session_state.master_answers.values())),
        }

        supabase = get_supabase()
        supabase.table("bcr_results").insert(submission_data).execute()
        return None

    except Exception as e:
        return (
            f"Supabase insert failed — {type(e).__name__}: {e}\n\n"
            f"Payload sent: {submission_data}"
        )


def write_results() -> tuple[int, int]:
    """
    Read every saved answer from master_answers (not from live widget state)
    and append them to RESULTS.csv.  Returns (correct_count, total_saved).

    master_answers is keyed by Question_ID and each value is already a
    complete record dict, so no grading logic is needed here.
    """
    records       = list(st.session_state.master_answers.values())
    correct_count = sum(1 for r in records if r.get("Is_Correct"))
    fieldnames    = ["PIN", "Course", "SLO", "Question_ID", "Student_Answer", "Is_Correct"]
    results_path_str = str(RESULTS_PATH)

    if not os.path.exists(results_path_str):
        with open(results_path_str, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    else:
        with open(results_path_str, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(records)

    return correct_count, len(records)


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------
# Streamlit reruns the entire script on every widget interaction.
# session_state persists values across those reruns within one browser session.
#
#   logged_in      — gates which screen is shown; set once at login, never
#                    reset during the session, so the user can't drift back
#                    to the login page by interacting with widgets.
#
#   pin            — stored so write_results() can tag CSV rows without
#                    passing it as a parameter through every call.
#
#   exam_questions — the randomly sampled list, generated exactly ONCE right
#                    after login and frozen here.  Without this, every radio
#                    click would re-run generate_exam() and scramble the exam.
#
#   submitted      — flips to True after the student submits; switches the
#                    view from exam → confirmation without re-grading.
# ---------------------------------------------------------------------------
defaults = {
    "logged_in":          False,
    "pin":                None,
    "exam_questions":     None,
    "submitted":          False,
    "current_page_index": 0,     # which course section the student is viewing
    "master_answers":     {},    # persists every answer across page navigation
    "supabase_error":     None,  # holds error string if Supabase push fails
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ---------------------------------------------------------------------------
# Page config  (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BCR Assessment",
    page_icon="📋",
    layout="centered",
)


# ===========================================================================
# SCREEN 1 — LOGIN
# ===========================================================================
if not st.session_state.logged_in:

    st.title("📋 Business Core Reflection")
    st.subheader("Student Assessment Portal")
    st.write("Enter your Xavier University email address to begin.")

    with st.form("login_form"):
        email_input = st.text_input(
            "Xavier Email Address",
            placeholder="username@xavier.edu",
        )
        login_btn = st.form_submit_button("Start Assessment", use_container_width=True)

    if login_btn:
        entered_email = str(email_input).strip().lower()
        roster        = load_student_roster()
        if entered_email in roster:
            all_questions = load_question_bank()
            missing       = parse_missing_courses(roster[entered_email])

            st.session_state.logged_in          = True
            st.session_state.pin                = entered_email
            st.session_state.exam_questions     = generate_exam(missing, all_questions)
            st.session_state.submitted          = False
            st.session_state.current_page_index = 0   # always start at section 1
            st.session_state.master_answers     = {}  # clear any previous session's answers
            st.rerun()                          # re-run immediately → exam screen
        else:
            st.error("❌ Email not found. Please verify your Xavier email address and try again.")


# ===========================================================================
# SCREEN 2 — EXAM
# ===========================================================================
elif not st.session_state.submitted:

    exam = st.session_state.exam_questions

    # ── Build stable ordered course list (preserving exam order) ───────────
    seen_courses: list[str] = []
    for q in exam:
        if q["Course"] not in seen_courses:
            seen_courses.append(q["Course"])

    total_pages = len(seen_courses)
    page_idx    = st.session_state.current_page_index
    # Guard: clamp in case exam changed size between sessions
    page_idx    = max(0, min(page_idx, total_pages - 1))

    current_course = seen_courses[page_idx]
    course_qs      = [q for q in exam if q["Course"] == current_course]
    is_last_page   = (page_idx == total_pages - 1)

    # ── Page header ──────────────────────────────────────────────────────────
    st.title("Business Core Reflection Assessment")
    st.caption(f"Logged in as: {st.session_state.pin}  •  Total questions: {len(exam)}")
    st.progress(
        (page_idx + 1) / total_pages,
        text=f"Section {page_idx + 1} of {total_pages}",
    )
    st.divider()

    st.header(f"Section: {format_course_label(current_course)}")
    st.caption(f"{len(course_qs)} question(s) in this section")

    # ── Questions for the current course only ─────────────────────────────
    # Calculate the global question offset so numbering is continuous
    questions_before = sum(
        len([q for q in exam if q["Course"] == seen_courses[i]])
        for i in range(page_idx)
    )

    for i, q in enumerate(course_qs):
        global_num = questions_before + i + 1

        options = []
        for letter in OPTION_LETTERS:
            text = q.get(f"Option_{letter}", "").strip()
            if text:
                options.append(f"{letter}: {escape_md(text)}")

        qid = q.get("Question_ID", "")
        st.markdown(f"**{global_num}. {escape_md(q.get('Question_Text', ''))}**")

        image_file = (q.get("Image_File", "") or "").strip()
        if image_file:
            try:
                st.image(str(IMAGES_DIR / image_file))
            except Exception:
                st.caption("_(image unavailable)_")

        st.radio(
            label="answer",
            options=options,
            key=qid,                           # bare Question_ID — widget state key
            index=None,
            label_visibility="collapsed",
            on_change=save_answer,             # fires on every selection change
            args=(                             # positional args passed to save_answer
                qid,
                q.get("Course", ""),
                q.get("SLO", ""),
                q.get("Correct_Answer", ""),
            ),
        )
        st.write("")

    st.divider()

    # ── Navigation buttons ────────────────────────────────────────────────
    col_prev, col_next = st.columns(2)

    with col_prev:
        if st.button(
            "◀  Previous",
            disabled=(page_idx == 0),
            use_container_width=True,
        ):
            st.session_state.current_page_index -= 1
            st.rerun()

    with col_next:
        if not is_last_page:
            if st.button("Next  ▶", type="primary", use_container_width=True):
                st.session_state.current_page_index += 1
                st.rerun()

    # ── Submit — visible only on the final section ────────────────────────
    if is_last_page:
        # Count unanswered from master_answers, not live widget state —
        # questions on earlier pages have no active widgets
        unanswered = sum(
            1 for q in exam
            if q.get("Question_ID", "") not in st.session_state.master_answers
        )
        if unanswered:
            st.warning(
                f"⚠️  {unanswered} question(s) unanswered across all sections. "
                "You may still submit."
            )

        if st.button("Submit Exam", type="primary", use_container_width=True):
            correct_count, _ = write_results()   # reads master_answers, not widget state
            err = push_to_supabase(
                correct_count,
                len(st.session_state.exam_questions),
            )
            st.session_state.supabase_error = err
            st.session_state.submitted = True
            st.rerun()


# ===========================================================================
# SCREEN 3 — CONFIRMATION
# ===========================================================================
else:
    # Re-compute score from master_answers — widget state is gone after submission
    correct = sum(
        1 for rec in st.session_state.master_answers.values()
        if rec.get("Is_Correct")
    )
    total = len(st.session_state.exam_questions)
    pct   = round(correct / total * 100) if total else 0

    st.title("✅ Assessment Submitted")
    st.success(
        "Your responses have been recorded successfully. "
        "Thank you for completing the Business Core Reflection Assessment.",
        icon="✅",
    )

    col1, col2 = st.columns(2)
    col1.metric("Questions Answered", total)
    col2.metric("Correct Responses", f"{correct} / {total} ({pct}%)")

    st.info(
        "Your results have been saved. You may safely close this window.",
        icon="💾",
    )

    if st.session_state.supabase_error:
        st.error(
            f"⚠️  Database sync failed (your CSV results are still saved).\n\n"
            f"{st.session_state.supabase_error}",
            icon="🛑",
        )
