"""Analysis 2b — for the high-dispersion named reference classes, tally how many
look like a citable dataset plausibly exists (sports seeds, election histories,
economic series, regulatory approvals) vs genuinely unquantifiable / bespoke classes.

This sizes the "route these to gap-fill base-rate research" idea: a dispersed class
that has a findable dataset is a good routing target; a dispersed bespoke class is not.

The classification is my judgment call, question by question, based on whether a
public source with the numerator/denominator exists (as demonstrated by the audit's
own verification pass, which found citable data for most of them).

Free/local only. Descriptive tally, no scoring.
"""

from collections import Counter

# Each entry: (post_id, class_label, dispersion_note, citable, source_type, comment)
# citable in {"yes","partial","no"}: yes = a public dataset with the num/denom exists and
# the audit or common knowledge confirms it; partial = a series exists but the exact
# reference-class slice needs assembly/judgment; no = genuinely bespoke / one-off / not a
# frequency a dataset would answer.
CLASSES = [
    # --- classes where >=3 models stated a rate on ~the same historical frequency ---
    (
        "41754",
        "annual leader-exit hazard, aged authoritarian / personalist regime",
        "1-3% actuarial vs 4-6% personalist vs blends",
        "yes",
        "poli-sci dataset",
        "Geddes/Wright/Frantz + Archigos + actuarial tables (audit G8 pinned it)",
    ),
    (
        "41835",
        "US shutdown per funding deadline, post-2010",
        "5-15% vs 15-25% vs 5-10% conditional",
        "yes",
        "gov record",
        "Wikipedia shutdown list; num/denom clean (audit G1)",
    ),
    (
        "41846",
        "successful coups/yr Africa 2020-25",
        "1.8 vs 1.9/yr — TIGHT (research-attributed)",
        "yes",
        "event dataset",
        "coup dataset, 1.83/yr exact (audit G2). Low dispersion BECAUSE sourced.",
    ),
    (
        "42116",
        "BW CDU-plurality rate (recent era vs long run)",
        "33% recent vs 80-90% longrun vs 95% poll-conversion",
        "yes",
        "election archive",
        "Wikipedia BW Landtag results; recency vs long-run is the whole dispersion (audit C4)",
    ),
    (
        "42120",
        "distressed/ULCC airline bankruptcy per window",
        "0.5% vs 5-8% vs 25-35% annual",
        "partial",
        "corp filings",
        "airline bankruptcy history exists but the ULCC-with-liquidity slice needs assembly",
    ),
    (
        "42242",
        "enwiki net article growth/month",
        "14-18k/mo — TIGHT; spike-prob <1% vs 3% vs 5%",
        "yes",
        "live stat series",
        "Wikipedia stats are directly countable; growth rate agreed, spike-prob is the judgment",
    ),
    (
        "42248",
        "WHR #1 incumbent-retention rate",
        "streak facts agree; retention 75-90%",
        "yes",
        "annual report",
        "WHR is public; Finland streak countable (audit implicit)",
    ),
    (
        "42438",
        "NCAA #1-overall-seed round-exit distribution",
        "F4 50% vs 54-58%; 3 conflicting full tables",
        "yes",
        "sports reference",
        "bracket/seed history fully public; three models gave three tables (audit C1)",
    ),
    (
        "42509",
        "unconditional Brent price-band occupancy",
        "band mass 10-25% on >=100; lognormal vs empirical",
        "yes",
        "EIA price series",
        "EIA daily Brent series is public; band frequencies computable",
    ),
    (
        "42514",
        "monthly Android Bulletin >=1 Critical rate",
        "40% vs 40% vs 45% — TIGHT",
        "yes",
        "vendor bulletins",
        "Android Security Bulletins are public monthly; near-agreement already",
    ),
    (
        "42646",
        "UNMISS renewal >=1-abstention rate",
        "30-40% all-missions vs 65% substantive vs 90-100% recent",
        "partial",
        "UNSC voting record",
        "UNSC voting records public but the reference-class definition (substantive vs technical) drives spread",
    ),
    (
        "42648",
        "ICJ decisions per 2-month window",
        "1-4 vs 2-3 vs 2.3-4.7 per window",
        "partial",
        "ICJ case list",
        "ICJ case list public but order-level per-year tally must be assembled (audit G3 medium-conf)",
    ),
    (
        "42800",
        "March-wins-Q1 rate, TSMC monthly revenue",
        "Mar 60% vs 55-60% — TIGHT and correct",
        "yes",
        "company IR",
        "TSMC IR monthly revenue fully public; models agreed AND were right (audit M8)",
    ),
    (
        "42805",
        "Bulgarian polling-leader-wins-plurality rate",
        "75-85% vs 70-80% vs 80%",
        "partial",
        "election archive",
        "Bulgarian election history public; the 'new-party-leader' slice needs judgment",
    ),
    (
        "42926",
        "US novel-flu case arrival rate",
        "0.27/wk 2026-YTD vs 1.3/wk 2025-H5 vs seasonal",
        "yes",
        "CDC FluView",
        "CDC FluView public; dispersion is which-window/seasonality, the real error (audit G7)",
    ),
    (
        "43131",
        "Anthropic flagship release cadence",
        "90-120d vs ~90d vs 17%/window",
        "yes",
        "release history",
        "release dates public/countable; recent cadence ~74d (audit G6) — sourcing fixes it",
    ),
    (
        "43614",
        "3-week generic-ballot drift (midterm)",
        "<1pt agreed; 2018/2022 anchors agreed — TIGHT",
        "yes",
        "polling aggregators",
        "538/RCP archives public; models near-identical (audit M4)",
    ),
    (
        "43652",
        "Armenia 2021 vote->seat conversion",
        "53.9%->66.4% agreed by 3 models — TIGHT",
        "yes",
        "election result",
        "single 2021 result, directly citable; agreement is high",
    ),
    (
        "43828",
        "Swiss popular-initiative pass rate",
        "10-12% overall agreed; SVP-slice 20-33%",
        "yes",
        "referendum archive",
        "Swiss initiative archive public (~11.5%); SVP-slice is the dispersion (audit C5)",
    ),
    (
        "43915",
        "FDA priority-review renal-surrogate approval rate",
        "7/7 ~85% vs 81% vs 75-80% — fairly TIGHT",
        "yes",
        "FDA approvals",
        "FDA approval record public; class is narrow but enumerable",
    ),
    (
        "43982",
        "AfD-leads-R1 eastern-runoff conversion rate",
        "30% vs 40% vs 45%",
        "yes",
        "election results",
        "eastern-German runoff results public; grok's 30% (3/10) was right (audit C3)",
    ),
    # --- classes that are genuinely hard to quantify from a dataset ---
    (
        "41672",
        "community aggregate beats a pre-selected expert (tournament)",
        "50% symmetry vs 65% vs 75%",
        "no",
        "n/a",
        "no clean dataset for 'CP beats THIS named individual over a season'; priors are structural guesses",
    ),
    (
        "41848",
        "multistate-AG settlement within a short window",
        "monthly hazard 1-5% vs 20% vs time-to-settle years",
        "partial",
        "legal dockets",
        "settlement timelines exist but 'this case in 22 days' is bespoke; hazard is a guess",
    ),
    (
        "41838",
        "Alphabet EMEA Q3->Q4 sequential uplift",
        "3-5% vs 5-9% vs 11-12% YoY (basis-mixed)",
        "yes",
        "SEC filings",
        "quarterly revenue public; the dispersion is as-reported vs constant-currency basis (audit M3)",
    ),
    (
        "41841",
        "US CPI year-over-year change magnitude",
        "±1-3 agreed; trend 75->65 agreed — TIGHT",
        "yes",
        "transparency.org",
        "CPI series public; models agreed and were right (audit M2)",
    ),
    (
        "42355",
        "10-day NQ-ES spread sigma / drift",
        "~1.5-1.8pp agreed; vols vary",
        "partial",
        "index data",
        "index vol/corr derivable but no clean free realized-vol series; models roughly agreed (audit M5)",
    ),
    (
        "43906",
        "2-week gold-vs-equity relative return",
        "~0/47% agreed — TIGHT but noisy",
        "partial",
        "index/commodity data",
        "computable in principle; no clean published stat, graded generously (audit M6)",
    ),
]


def main() -> None:

    print(f"=== 2b. NAMED-CLASS CITABILITY TALLY (n={len(CLASSES)} classes) ===\n")
    cite = Counter(c[3] for c in CLASSES)
    print("citable-dataset-plausibly-exists:")
    print(f"  yes     = {cite['yes']:2d}  (public dataset with num/denom, audit-confirmed or countable)")
    print(f"  partial = {cite['partial']:2d}  (series exists but the exact class-slice needs assembly/judgment)")
    print(f"  no      = {cite['no']:2d}  (genuinely bespoke / no frequency a dataset would answer)")
    print(
        f"  --> routable to a base-rate lookup (yes+partial) = {cite['yes'] + cite['partial']}/{len(CLASSES)} "
        f"({100 * (cite['yes'] + cite['partial']) / len(CLASSES):.0f}%)"
    )
    print()
    print(f"{'pid':6s} {'cite':8s} {'source':16s} class")
    for pid, label, disp, citable, src, comment in sorted(CLASSES, key=lambda c: (c[3], c[0])):
        print(f"{pid:6s} {citable:8s} {src[:16]:16s} {label}")  # noqa: HARNESS-SCAN-EXEMPT-subsampling  # string display truncation
        print(f"       dispersion: {disp}")


if __name__ == "__main__":
    main()
