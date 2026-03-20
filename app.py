"""
app.py — Business Core Reflection Assessment
---------------------------------------------
Run with:  streamlit run app.py
"""

import csv
import os
import random
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths (all relative to this file so the app works from any working dir)
# ---------------------------------------------------------------------------
BASE_DIR          = Path(__file__).parent
ACCESS_CODES_PATH = BASE_DIR / "ACCESS_CODES.csv"
QUESTION_BANK_PATH= BASE_DIR / "QUESTION_BANK.csv"
RESULTS_PATH      = BASE_DIR / "RESULTS.csv"

QUESTIONS_PER_COURSE = 7
OPTION_LETTERS       = ["A", "B", "C", "D", "E"]


# ---------------------------------------------------------------------------
# Data helpers  (cached so CSVs are only read once per Streamlit process)
# ---------------------------------------------------------------------------
@st.cache_data
def load_access_codes() -> dict[str, str]:
    """Returns {PIN: Missing_Courses_string}.
    PIN keys are explicitly cast to str and stripped so the type is
    guaranteed regardless of how the CSV was saved (e.g. pandas may
    write integer columns without quotes, causing a type mismatch on
    lookup if the key is not normalised here).
    """
    codes = {}
    with open(ACCESS_CODES_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            codes[str(row["PIN"]).strip()] = row["Missing_Courses"].strip()
    return codes


@st.cache_data
def load_question_bank() -> list[dict]:
    with open(QUESTION_BANK_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_missing_courses(missing_str: str) -> set[str]:
    """
    Convert the Missing_Courses string from ACCESS_CODES.csv into a normalised
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
    Build a personalised exam using strict per-course sampling:

    1. Load the full question bank into a DataFrame.
    2. Filter OUT every course whose code appears in missing_courses
       (those are courses the student has NOT yet completed and should
       not be assessed on).
    3. Group the remaining rows by Course and call .sample() on each
       group to draw exactly QUESTIONS_PER_COURSE questions.
       If a course has fewer questions than the target, all of them are
       returned without raising an error.
    4. Convert back to a list of plain dicts for the rest of the app.
    """
    df = pd.DataFrame(all_questions)

    # Boolean mask: True for every row whose course IS completed
    completed_mask = ~df["Course"].isin(missing_courses)
    df_completed   = df[completed_mask]

    if df_completed.empty:
        return []

    # Sample up to QUESTIONS_PER_COURSE rows per course group
    sampled = (
        df_completed
        .groupby("Course", group_keys=False)
        .apply(lambda g: g.sample(n=min(QUESTIONS_PER_COURSE, len(g))),
               include_groups=False)
    )

    # include_groups=False drops the "Course" column from the lambda's view;
    # restore it by re-merging with the original Course values via the index.
    sampled["Course"] = df_completed.loc[sampled.index, "Course"]

    return sampled.to_dict("records")


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
    st.write("Enter the 6-digit access code provided to you.")

    with st.form("login_form"):
        pin_input = st.text_input(
            "6-Digit Access Code",
            max_chars=6,
            placeholder="______",
            type="password",
        )
        login_btn = st.form_submit_button("Start Assessment", use_container_width=True)

    if login_btn:
        entered_pin   = str(pin_input).strip()
        access_codes  = load_access_codes()
        if entered_pin in access_codes:
            all_questions = load_question_bank()
            missing       = parse_missing_courses(access_codes[entered_pin])

            st.session_state.logged_in          = True
            st.session_state.pin                = entered_pin
            st.session_state.exam_questions     = generate_exam(missing, all_questions)
            st.session_state.submitted          = False
            st.session_state.current_page_index = 0   # always start at section 1
            st.session_state.master_answers     = {}  # clear any previous session's answers
            st.rerun()                          # re-run immediately → exam screen
        else:
            st.error("❌ Invalid access code. Please try again.")


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
    st.caption(f"Access code: {st.session_state.pin}  •  Total questions: {len(exam)}")
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
            write_results()                    # reads master_answers, not widget state
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
