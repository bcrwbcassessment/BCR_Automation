"""
build_interactive_report.py — BCR Mega Dashboard (Xavier Brand)
---------------------------------------------------------------
Run with:  python3 build_interactive_report.py

Input sources
  Supabase bcr_results table — live student responses (fetched at run time)
  LOCAL_DEMOGRAPHICS.csv     — Email + GPA / Major / Transfer / Class (no names, local)
  HISTORICAL_RESULTS.csv     — all prior semesters, accumulated (local)

Output
  Final_Committee_Report.html — single offline HTML file

Architecture
  Section A and B are exported as two separate Vega-Lite JSON specs and
  rendered by independent vegaEmbed calls in the HTML.  This lets the custom
  HTML/CSS wrapper place each section in its own Xavier-branded card without
  relying on Altair's vertical-concat (&) layout.

Offline embedding
  Altair serialises each DataFrame as inline JSON in the spec.  The three
  Vega JS libs are loaded from CDN once; all filter/aggregation logic runs
  purely client-side in the browser — no server required after first load.
"""

import re
import sys
import warnings
from pathlib import Path

import altair as alt
import pandas as pd

from supabase_fetch import fetch_results_df, save_backup, report_missing_students

# Altair 6 fires a false-positive UserWarning when the same param objects
# appear in multiple sub-charts composed with | or &.  The deduplication is
# correct and desired.  See github.com/vega/altair/issues/3891.
warnings.filterwarnings(
    "ignore",
    message="Automatically deduplicated selection parameter",
    category=UserWarning,
)

# ---------------------------------------------------------------------------
# Xavier University brand palette
# ---------------------------------------------------------------------------
XAVIER_NAVY   = "#0C2340"   # Primary Blue  — headers, bar fills, axis titles
XAVIER_BLUE   = "#0099CC"   # Secondary Blue — trend lines, accent borders
XAVIER_SILVER = "#9EA2A2"   # Silver/Grey    — subtitles, muted labels
XAVIER_BG     = "#F4F7F6"   # Page background

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE_DIR        = Path(__file__).parent
DEMO_PATH       = BASE_DIR / "LOCAL_DEMOGRAPHICS.csv"
HISTORICAL_PATH = BASE_DIR / "HISTORICAL_RESULTS.csv"
OUTPUT_PATH     = BASE_DIR / "Final_Committee_Report.html"

HIST_COLS     = ["PIN", "Course", "SLO", "Question_ID",
                 "Student_Answer", "Is_Correct", "Semester"]
GPA_BIN_ORDER = ["<2.0", "2.0–2.4", "2.5–2.9", "3.0–3.4", "3.5–3.9", "4.0"]

alt.data_transformers.disable_max_rows()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_course(code: str) -> str:
    """'ACCT200' → 'ACCT 200'"""
    m = re.match(r"([A-Z]+)(\d+)$", str(code).strip())
    return f"{m.group(1)} {m.group(2)}" if m else str(code)


def to_int_correct(series: pd.Series) -> pd.Series:
    """Convert 'True'/'False' or bool column to int (1/0) for Altair mean()."""
    if series.dtype == bool:
        return series.astype(int)
    return (series.str.strip().str.lower() == "true").astype(int)


def make_dropdown(field: str, label: str, options: list) -> alt.selection_point:
    """
    Selection param bound to a <select> widget.
    value=None + empty=True → 'All' option passes every row through the filter.
    """
    return alt.selection_point(
        fields=[field],
        bind=alt.binding_select(
            options=[None] + options,
            labels=["All"] + options,
            name=label,
        ),
        value=None,
        empty=True,
    )


# ── Shared Altair configuration applied to every exported spec ───────────────
SHARED_CONFIG = dict(
    view=dict(strokeWidth=0),
    axis=dict(
        grid=False,
        labelFontSize=11,
        titleFontSize=12,
        titleFontWeight="bold",
        titleColor=XAVIER_NAVY,
        labelColor="#444444",
        domainColor=XAVIER_SILVER,
        tickColor=XAVIER_SILVER,
    ),
    title=dict(
        fontSize=14,
        fontWeight="bold",
        color=XAVIER_NAVY,
        subtitleColor=XAVIER_SILVER,
        subtitleFontSize=11,
    ),
    legend=dict(
        titleFontSize=11,
        labelFontSize=11,
        titleColor=XAVIER_NAVY,
        labelColor="#444444",
    ),
)


def apply_config(chart):
    """Apply all shared configure_* calls to a top-level Altair chart."""
    return (
        chart
        .configure_view(**SHARED_CONFIG["view"])
        .configure_axis(**SHARED_CONFIG["axis"])
        .configure_title(**SHARED_CONFIG["title"])
        .configure_legend(**SHARED_CONFIG["legend"])
    )


# ===========================================================================
# PART 1 — FETCH LIVE DATA FROM SUPABASE
# ===========================================================================
raw_results = fetch_results_df()

# Hard-copy backup written to disk on every run
save_backup(raw_results)

# Private console + local-file report of students who haven't submitted yet
# (reads STUDENT_ROSTER.csv; output never reaches the HTML dashboard)
report_missing_students(set(raw_results["PIN"].unique()))

current_merged = None

if not raw_results.empty:
    if not DEMO_PATH.exists():
        print(f"ERROR: {DEMO_PATH.name} not found. Run generate_access_codes.py first.")
        sys.exit(1)

    demo_df        = pd.read_csv(DEMO_PATH, dtype=str)
    current_merged = raw_results.merge(demo_df, left_on="PIN", right_on="Email", how="inner")

    if current_merged.empty:
        print(
            "WARNING: Supabase emails and LOCAL_DEMOGRAPHICS.csv share no matches.\n"
            "Both must come from the same generate_access_codes.py run.\n"
            "Section A will be skipped."
        )
        current_merged = None
    else:
        # Anonymise: assign a sequential integer Student_ID so distinct() still
        # works in Altair tooltips/stat cards, then drop the email columns so
        # no PII is embedded in the HTML output.
        _email_map = {e: i + 1 for i, e in enumerate(sorted(current_merged["PIN"].unique()))}
        current_merged["Student_ID"] = current_merged["PIN"].map(_email_map)
        current_merged.drop(columns=["PIN", "Email"], inplace=True)

        current_merged["Is_Correct"]   = to_int_correct(current_merged["Is_Correct"])
        current_merged["Course_Label"] = current_merged["Course"].apply(fmt_course)
        print(
            f"Current semester: {current_merged['Student_ID'].nunique()} students · "
            f"{len(current_merged)} responses · "
            f"{current_merged['Course_Label'].nunique()} courses"
        )


# ===========================================================================
# PART 2 — ARCHIVE TO HISTORICAL_RESULTS.CSV  (optional — press Enter to skip)
# ===========================================================================
# Archiving appends the current Supabase data to the local longitudinal file
# so it feeds into the Section B trend charts below.  Skipping is safe — the
# current-semester charts still render; only the trend charts are affected.
# ---------------------------------------------------------------------------
if not raw_results.empty:
    if not HISTORICAL_PATH.exists():
        pd.DataFrame(columns=HIST_COLS).to_csv(HISTORICAL_PATH, index=False)
        print(f"Created empty {HISTORICAL_PATH.name}")

    semester = input(
        "\nArchive this semester's data to HISTORICAL_RESULTS.csv?\n"
        "Enter semester name (e.g., Spring 2026) or press Enter to skip: "
    ).strip()

    if semester:
        archive_df            = raw_results.copy()
        archive_df["Semester"] = semester
        archive_df             = archive_df[[c for c in HIST_COLS if c in archive_df.columns]]

        write_header = HISTORICAL_PATH.stat().st_size == 0
        archive_df.to_csv(HISTORICAL_PATH, mode="a", header=write_header, index=False)
        print(f"Archived {len(archive_df)} rows → {HISTORICAL_PATH.name}")
        print("(Source data remains in Supabase — no local file renamed.)")
    else:
        print("Archiving skipped.")


# ===========================================================================
# PART 3 — HISTORICAL DATA
# ===========================================================================
hist_df = None
if HISTORICAL_PATH.exists():
    hist_df = pd.read_csv(
        HISTORICAL_PATH,
        dtype=str,
        usecols=lambda c: c in set(HIST_COLS),   # guard against stray columns
    )
    if hist_df.empty:
        hist_df = None
    else:
        hist_df["Is_Correct"]   = to_int_correct(hist_df["Is_Correct"])
        hist_df["Course_Label"] = hist_df["Course"].apply(fmt_course)
        semester_order          = list(dict.fromkeys(hist_df["Semester"]))
        print(
            f"Historical: {hist_df['PIN'].nunique()} students · "
            f"{len(hist_df)} responses · "
            f"{len(semester_order)} semester(s): {semester_order}"
        )

if current_merged is None and hist_df is None:
    print("No data available. Exiting.")
    sys.exit(0)


# ===========================================================================
# PART 4 — SECTION A: CURRENT-SEMESTER INTERACTIVE CHARTS
# ===========================================================================
spec_a_json = None

if current_merged is not None:
    # ── Dropdown option lists ────────────────────────────────────────────────
    sorted_majors    = sorted(current_merged["Primary_Major"].unique())
    gpa_bins_present = [b for b in GPA_BIN_ORDER if b in current_merged["GPA_Bin"].unique()]
    xfer_options     = sorted(current_merged["Transfer_Status"].unique())
    class_options    = sorted(current_merged["Class_Standing"].unique())

    # ── Four dropdown selectors ──────────────────────────────────────────────
    major_sel = make_dropdown("Primary_Major",   "Student Major (Filter by Academic Program): ", sorted_majors)
    gpa_sel   = make_dropdown("GPA_Bin",         "GPA Tier (Filter by Overall GPA): ",          gpa_bins_present)
    xfer_sel  = make_dropdown("Transfer_Status", "Admission Type (Native vs. Transfer): ",      xfer_options)
    class_sel = make_dropdown("Class_Standing",  "Class Standing (Junior vs. Senior): ",        class_options)

    # ── Base: four demographic params + filters, inherited by both charts ────
    base_a = (
        alt.Chart(current_merged)
        .add_params(major_sel, gpa_sel, xfer_sel, class_sel)
        .transform_filter(major_sel)
        .transform_filter(gpa_sel)
        .transform_filter(xfer_sel)
        .transform_filter(class_sel)
    )

    # ── Course-isolation selector (SLO chart only) ───────────────────────────
    #
    # This selector is intentionally NOT added to base_a.  Adding it there
    # would also filter the course bar chart, hiding all courses except the
    # selected one.  Instead it is layered onto slo_chart alone so the two
    # charts serve distinct analytical purposes:
    #
    #   course_chart  ← demographic filters only   (all courses side-by-side)
    #   slo_chart     ← demographic + course filter (one course's LOs at a time)
    #
    # value=sorted_courses[0] ensures the chart is never blank on first load.
    # empty=False means a non-matching selection shows nothing rather than
    # everything — avoids the misleading "all LOs mixed" state.
    sorted_courses  = sorted(current_merged["Course_Label"].unique())
    course_slo_sel  = alt.selection_point(
        fields=["Course_Label"],
        bind=alt.binding_select(
            options=sorted_courses,
            name="Target Course (Select to reveal SLOs and specific Questions): ",
        ),
        value=sorted_courses[0],
        empty=False,
    )

    # ── Bar chart 1: Average Score by Course ────────────────────────────────
    # color= is set on mark_bar (fixed Xavier navy), NOT as an encoding channel.
    # This means no colour legend and all bars are uniform — clean, uncluttered.
    course_chart = (
        base_a
        .mark_bar(
            color=XAVIER_NAVY,
            cornerRadiusTopLeft=4,
            cornerRadiusTopRight=4,
        )
        .encode(
            x=alt.X(
                "Course_Label:N",
                sort=alt.EncodingSortField(
                    field="Is_Correct", op="mean", order="descending"
                ),
                title="Course",
                axis=alt.Axis(labelAngle=-35),
            ),
            y=alt.Y(
                "mean(Is_Correct):Q",
                title="Average Score",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            tooltip=[
                alt.Tooltip("Course_Label:N",     title="Course"),
                alt.Tooltip("mean(Is_Correct):Q", title="Avg Score",  format=".1%"),
                alt.Tooltip("count():Q",          title="Responses"),
                alt.Tooltip("distinct(Student_ID):Q",    title="Students"),
            ],
        )
        .properties(
            title="Average Score by Course",
            width=460,
            height=300,
        )
    )

    # ── Bar chart 2: Average Score by SLO (single course) ────────────────────
    #
    # Chain course_slo_sel onto base_a here — not on base_a itself — so the
    # course filter applies only to this chart.  The demographic filters from
    # base_a (major, GPA, transfer, class) are still active via inheritance.
    slo_base  = (
        base_a
        .add_params(course_slo_sel)
        .transform_filter(course_slo_sel)
    )
    slo_chart = (
        slo_base
        .mark_bar(
            color=XAVIER_NAVY,
            cornerRadiusTopLeft=4,
            cornerRadiusTopRight=4,
        )
        .encode(
            x=alt.X(
                "SLO:N",
                title="Learning Outcome",
                axis=alt.Axis(labelAngle=0),
                sort="x",
            ),
            y=alt.Y(
                "mean(Is_Correct):Q",
                title="Average Score",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            tooltip=[
                alt.Tooltip("Course_Label:N",     title="Course"),
                alt.Tooltip("SLO:N",              title="Learning Outcome"),
                alt.Tooltip("mean(Is_Correct):Q", title="Avg Score",  format=".1%"),
                alt.Tooltip("count():Q",          title="Responses"),
            ],
        )
        .properties(
            title=alt.Title(
                "Average Score by SLO",
                subtitle="Filtered to the course selected above.",
            ),
            width=300,
            height=180,
        )
    )

    # ── Bar chart 3: Item Analysis — Average Score by Question ───────────────
    #
    # Built on slo_base so it inherits BOTH the four demographic params/filters
    # from base_a AND the course-isolation filter from course_slo_sel.
    # When the committee changes the course dropdown, all three drill-down
    # panels (SLO breakdown, item analysis) update simultaneously.
    question_chart = (
        slo_base
        .mark_bar(
            color=XAVIER_NAVY,
            cornerRadiusTopLeft=4,
            cornerRadiusTopRight=4,
        )
        .encode(
            x=alt.X(
                "Question_ID:N",
                title="Question",
                sort="x",
                axis=alt.Axis(labelAngle=-45, labelLimit=140),
            ),
            y=alt.Y(
                "mean(Is_Correct):Q",
                title="Average Score",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            tooltip=[
                alt.Tooltip("Course_Label:N",     title="Course"),
                alt.Tooltip("SLO:N",              title="Learning Outcome"),
                alt.Tooltip("Question_ID:N",      title="Question ID"),
                alt.Tooltip("mean(Is_Correct):Q", title="Avg Score",  format=".1%"),
                alt.Tooltip("count():Q",          title="Responses"),
                alt.Tooltip("distinct(Student_ID):Q",    title="Students"),
            ],
        )
        .properties(
            title=alt.Title(
                "Item Analysis",
                subtitle="Avg score per question for the selected course.",
            ),
            width=300,
            height=180,
        )
    )

    # ── Executive summary: large stat cards + participation by major ─────────
    #
    # All three charts are built on base_a so they inherit the four demographic
    # params and filters.  Vega-Lite deduplicates the params at the top level,
    # meaning the dropdowns appear once and drive every chart in the spec.
    #
    # Stat cards use transform_aggregate to collapse the filtered rows to a
    # single summary value, then mark_text with alt.value() for pixel-exact
    # centring inside a fixed-size chart area.

    stat_n = (
        base_a
        .transform_aggregate(n="distinct(Student_ID)")
        .mark_text(
            fontSize=46,
            fontWeight="bold",
            color=XAVIER_NAVY,
            align="center",
            baseline="middle",
        )
        .encode(
            text=alt.Text("n:Q", format=","),
            x=alt.value(95),   # centre of width=190
            y=alt.value(48),   # centre of height=96
        )
        .properties(
            title=alt.Title(
                "Students (N)",
                fontSize=12,
                color=XAVIER_SILVER,
                anchor="middle",
            ),
            width=190,
            height=96,
        )
    )

    stat_avg = (
        base_a
        .transform_aggregate(avg="mean(Is_Correct)")
        .mark_text(
            fontSize=46,
            fontWeight="bold",
            color=XAVIER_NAVY,
            align="center",
            baseline="middle",
        )
        .encode(
            text=alt.Text("avg:Q", format=".1%"),
            x=alt.value(95),
            y=alt.value(48),
        )
        .properties(
            title=alt.Title(
                "Overall Avg Score",
                fontSize=12,
                color=XAVIER_SILVER,
                anchor="middle",
            ),
            width=190,
            height=96,
        )
    )

    major_bar = (
        base_a
        .transform_aggregate(
            student_count="distinct(Student_ID)",
            groupby=["Primary_Major"],
        )
        .mark_bar(
            color=XAVIER_NAVY,
            cornerRadiusTopRight=3,
            cornerRadiusBottomRight=3,
        )
        .encode(
            x=alt.X(
                "student_count:Q",
                title="Students",
                axis=alt.Axis(format="d", tickMinStep=1),
            ),
            y=alt.Y("Primary_Major:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("Primary_Major:N", title="Major"),
                alt.Tooltip("student_count:Q", title="Students"),
            ],
        )
        .properties(
            title=alt.Title(
                "Participation by Major",
                subtitle="Updates with filter selections. Count = distinct students.",
            ),
            width=440,
            height=max(150, len(sorted_majors) * 22),
        )
    )

    summary_row = stat_n | stat_avg | major_bar

    # ── Assemble: executive summary on top, main charts below ────────────────
    # Layout: course overview (left) | slo breakdown stacked above item analysis (right)
    # The slo_chart & question_chart vconcat keeps the drill-down panels
    # visually paired without making the row too wide for most screens.
    drill_down_col  = slo_chart & question_chart
    section_a_chart = apply_config(summary_row & (course_chart | drill_down_col))
    spec_a_json     = section_a_chart.to_json()


# ===========================================================================
# PART 5 — SECTION B: LONGITUDINAL TREND CHARTS
# ===========================================================================
spec_b_json = None

if hist_df is not None:
    semester_order = list(dict.fromkeys(hist_df["Semester"]))

    # ── Pre-aggregate in pandas (keeps embedded JSON small) ─────────────────
    overall_hist = (
        hist_df.groupby("Semester", sort=False)["Is_Correct"]
        .mean().mul(100).round(1)
        .reindex(semester_order)
        .reset_index()
        .rename(columns={"Is_Correct": "Avg_Score"})
    )

    all_courses = sorted(hist_df["Course_Label"].unique())
    course_hist = (
        hist_df.groupby(["Semester", "Course_Label"], sort=False)["Is_Correct"]
        .mean().mul(100).round(1)
        .reset_index()
        .rename(columns={"Is_Correct": "Avg_Score"})
        .set_index(["Semester", "Course_Label"])
        .reindex(pd.MultiIndex.from_product(
            [semester_order, all_courses], names=["Semester", "Course_Label"]
        ))
        .reset_index()
    )

    # ── Line chart 1: Overall trend ──────────────────────────────────────────
    # Xavier Secondary Blue (#0099CC) for the line and points.
    # Layer: line + points + text labels above each point.
    _overall_base = alt.Chart(overall_hist).encode(
        x=alt.X(
            "Semester:N",
            sort=semester_order,
            title="Semester",
            axis=alt.Axis(labelAngle=-25),
        ),
        y=alt.Y(
            "Avg_Score:Q",
            title="Average Score (%)",
            scale=alt.Scale(domain=[0, 100]),
        ),
    )

    overall_chart = (
        _overall_base.mark_line(color=XAVIER_BLUE, strokeWidth=3)
        + _overall_base.mark_point(color=XAVIER_BLUE, size=90, filled=True)
        + _overall_base.mark_text(
            color=XAVIER_NAVY, dy=-14, fontSize=11, fontWeight="bold"
          ).encode(text=alt.Text("Avg_Score:Q", format=".1f"))
    ).properties(
        title="Overall Average Score by Semester",
        width=380,
        height=280,
    ).encode(
        tooltip=[
            alt.Tooltip("Semester:N",  title="Semester"),
            alt.Tooltip("Avg_Score:Q", title="Avg Score (%)", format=".1f"),
        ]
    )

    # ── Line chart 2: Per-course multi-line ──────────────────────────────────
    course_trend_chart = (
        alt.Chart(course_hist)
        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=50))
        .encode(
            x=alt.X(
                "Semester:N",
                sort=semester_order,
                title="Semester",
                axis=alt.Axis(labelAngle=-25),
            ),
            y=alt.Y(
                "Avg_Score:Q",
                title="Average Score (%)",
                scale=alt.Scale(domain=[0, 100]),
            ),
            color=alt.Color(
                "Course_Label:N",
                title="Course",
                scale=alt.Scale(scheme="tableau20"),
            ),
            tooltip=[
                alt.Tooltip("Course_Label:N", title="Course"),
                alt.Tooltip("Semester:N",     title="Semester"),
                alt.Tooltip("Avg_Score:Q",    title="Avg Score (%)", format=".1f"),
            ],
        )
        .properties(
            title="Average Score by Course & Semester",
            width=480,
            height=280,
        )
    )

    # ── Horizontal concat → apply brand config → serialise ──────────────────
    section_b_chart = apply_config(overall_chart | course_trend_chart)
    spec_b_json     = section_b_chart.to_json()


# ===========================================================================
# PART 6 — ASSEMBLE BRANDED HTML
# ===========================================================================

# ── Extract exact CDN <script> tags from Altair (always version-correct) ─────
# Altair's to_html() emits e.g.:
#   <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/vega@6">
# We capture the full tag so type= and version pins are preserved exactly.
_probe      = alt.Chart(pd.DataFrame({"x": [1]})).mark_point().encode(x="x:Q").to_html()
_cdn_tags   = re.findall(r'<script[^>]+cdn\.jsdelivr\.net[^>]*></script>', _probe)
cdn_scripts = "\n  ".join(_cdn_tags)

# ── Build conditional HTML blocks ────────────────────────────────────────────
section_a_block = ""
if spec_a_json is not None:
    section_a_block = f"""
    <section class="section-block">
      <h2 class="section-title">Current Semester — Interactive View</h2>
      <p class="section-desc">
        The top row shows total students (N), overall average score, and participation
        by major — all three update instantly when a filter changes.
        Use the demographic dropdowns to filter every chart by Major, GPA Tier,
        Transfer Status, and Class Standing.
        Use the course dropdown to drill into SLO-level and individual question
        (item analysis) performance for any specific course.
      </p>
      <div class="card" id="card-a">
        <div id="chart-a"></div>
      </div>
    </section>"""

section_b_block = ""
if spec_b_json is not None:
    section_b_block = f"""
    <section class="section-block">
      <h2 class="section-title">Longitudinal Trends</h2>
      <p class="section-desc">
        Historical performance across all recorded semesters
        from HISTORICAL_RESULTS.csv.
      </p>
      <div class="card" id="card-b">
        <div id="chart-b"></div>
      </div>
    </section>"""

# ── JavaScript render calls ──────────────────────────────────────────────────
js_lines = []
if spec_a_json is not None:
    js_lines.append(
        f"  vegaEmbed('#chart-a', {spec_a_json}, "
        f"{{actions: false, renderer: 'svg'}}).catch(console.error);"
    )
if spec_b_json is not None:
    js_lines.append(
        f"  vegaEmbed('#chart-b', {spec_b_json}, "
        f"{{actions: false, renderer: 'svg'}}).catch(console.error);"
    )
js_block = "\n".join(js_lines)

# ── Full HTML page ────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>BCR Committee Report — Xavier WCB Assessment</title>

  {cdn_scripts}

  <style>
    /* ── Reset ──────────────────────────────────────────────────────── */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    /* ── Base ───────────────────────────────────────────────────────── */
    body {{
      font-family: "Segoe UI", Arial, sans-serif;
      background: {XAVIER_BG};
      color: {XAVIER_NAVY};
      padding-top: 72px;       /* clears fixed header */
      padding-bottom: 56px;
    }}

    /* ── Fixed header ───────────────────────────────────────────────── */
    .site-header {{
      position: fixed;
      top: 0; left: 0; right: 0;
      height: 64px;
      background: {XAVIER_NAVY};
      display: flex;
      align-items: center;
      padding: 0 32px;
      gap: 0;
      z-index: 200;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.35);
    }}
    .site-header .brand {{
      color: #ffffff;
      font-size: 1.1rem;
      font-weight: 700;
      letter-spacing: 0.025em;
      white-space: nowrap;
    }}
    .site-header .divider {{
      width: 1px;
      height: 28px;
      background: {XAVIER_SILVER};
      margin: 0 18px;
      flex-shrink: 0;
    }}
    .site-header .sub {{
      color: {XAVIER_SILVER};
      font-size: 0.85rem;
      letter-spacing: 0.02em;
    }}

    /* ── Page layout ────────────────────────────────────────────────── */
    main {{
      max-width: 1300px;
      margin: 0 auto;
      padding: 32px 24px 0;
    }}

    /* ── Section chrome ─────────────────────────────────────────────── */
    .section-block {{
      margin-bottom: 40px;
    }}
    .section-title {{
      font-size: 1.25rem;
      font-weight: 700;
      color: {XAVIER_NAVY};
      border-left: 4px solid {XAVIER_BLUE};
      padding-left: 12px;
      margin-bottom: 6px;
    }}
    .section-desc {{
      font-size: 0.85rem;
      color: {XAVIER_SILVER};
      padding-left: 16px;
      margin-bottom: 18px;
      line-height: 1.5;
    }}

    /* ── Chart card ─────────────────────────────────────────────────── */
    .card {{
      background: #ffffff;
      border-radius: 8px;
      box-shadow: 0 2px 12px rgba(12, 35, 64, 0.09);
      padding: 24px 28px;
      overflow-x: auto;       /* horizontal scroll if viewport is narrow */
    }}

    /* ── Intro paragraph ────────────────────────────────────────────── */
    .page-intro {{
      background: #ffffff;
      border-left: 4px solid {XAVIER_BLUE};
      border-radius: 0 6px 6px 0;
      padding: 14px 20px;
      margin-bottom: 32px;
      font-size: 0.95rem;
      color: #333333;
      line-height: 1.65;
      box-shadow: 0 1px 4px rgba(12, 35, 64, 0.06);
    }}

    /* ── Vega-Embed: control panel container (.vega-bindings) ────────── */
    /* Vega-Embed renders all bound widgets inside .vega-bindings.        */
    .vega-embed {{
      width: 100%;
    }}
    .vega-embed .vega-bindings {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px 0;
      padding: 15px 18px;
      margin-bottom: 20px;
      background: {XAVIER_BG};
      border-radius: 6px;
      border: 1px solid #e2e6ea;
    }}

    /* ── Individual dropdown wrapper (.vega-bind) ────────────────────── */
    .vega-embed .vega-bind {{
      display: flex;
      align-items: center;
      margin-right: 20px;
    }}

    /* ── Dropdown label (.vega-bind-name) ────────────────────────────── */
    .vega-embed .vega-bind-name {{
      font-size: 0.78rem;
      font-weight: 600;
      color: {XAVIER_NAVY};
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-right: 6px;
      white-space: nowrap;
    }}

    /* ── <select> element ────────────────────────────────────────────── */
    .vega-embed select {{
      border: 1px solid #c8cdd4;
      border-radius: 4px;
      padding: 6px 10px;
      font-size: 0.85rem;
      font-family: "Segoe UI", Arial, sans-serif;
      color: {XAVIER_NAVY};
      background: #ffffff;
      cursor: pointer;
      min-width: 140px;
      transition: border-color 0.15s;
    }}
    .vega-embed select:hover {{
      border-color: {XAVIER_BLUE};
    }}
    .vega-embed select:focus {{
      outline: 2px solid {XAVIER_BLUE};
      outline-offset: 1px;
      border-color: {XAVIER_BLUE};
    }}

    /* ── Footer ─────────────────────────────────────────────────────── */
    footer {{
      text-align: center;
      font-size: 0.75rem;
      color: {XAVIER_SILVER};
      margin-top: 48px;
      padding: 0 24px;
      line-height: 1.6;
    }}
    footer a {{
      color: {XAVIER_BLUE};
      text-decoration: none;
    }}
  </style>
</head>

<body>

  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <!-- FIXED HEADER                                                        -->
  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <header class="site-header">
    <span class="brand">Xavier University</span>
    <span class="divider"></span>
    <span class="sub">Williams College of Business &mdash; BCR Committee Report</span>
  </header>

  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <!-- MAIN CONTENT                                                        -->
  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <main>
    <p class="page-intro">
      Welcome to the <strong>Business Core Reflection (BCR) Assessment Dashboard</strong>.
      Use the control panel below to filter current semester performance by student
      demographics, or scroll down to view longitudinal trends.
    </p>
    {section_a_block}
    {section_b_block}
  </main>

  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <!-- FOOTER                                                              -->
  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <footer>
    <p>
      Generated by <code>build_interactive_report.py</code> &bull;
      Xavier WCB Assessment Committee &bull;
      Data embedded inline &mdash; no server required after first CDN load.
    </p>
  </footer>

  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <!-- VEGA-EMBED RENDER CALLS                                             -->
  <!-- ═══════════════════════════════════════════════════════════════════ -->
  <script>
{js_block}
  </script>

</body>
</html>"""

OUTPUT_PATH.write_text(html, encoding="utf-8")
size_kb = OUTPUT_PATH.stat().st_size // 1024
print(f"\nDashboard written → {OUTPUT_PATH.name}  ({size_kb} KB)")
print("Open in any modern browser — CDN loads once, then fully offline.")
