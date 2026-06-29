from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev


CHECKPOINT_DIR = Path("outputs/checkpoints")
SUMMARY_DIR = Path("outputs/paper/beauty2013")


FAMILY_SPECS = [
    {
        "family": "baseline",
        "label": "PyTorch SASRec baseline",
        "patterns": [
            "beauty2013_paper_v1_baseline.json",
            "beauty2013_paper_v46_baseline_seed*.json",
        ],
    },
    {
        "family": "source075_w15",
        "label": "Metadata + source text (source075_w15)",
        "patterns": [
            "beauty2013_metadata_v16_source075_w15_full_tuned.json",
            "beauty2013_metadata_v19_source075_w15_seed*.json",
        ],
    },
    {
        "family": "source075_w15_all",
        "label": "Metadata + source text all observed current-code seeds",
        "patterns": [
            "beauty2013_metadata_v16_source075_w15_full_tuned.json",
            "beauty2013_metadata_v19_source075_w15_seed*.json",
            "beauty2013_metadata_v49*_source075_w15_currentcode_seed*.json",
        ],
    },
    {
        "family": "category_profile_m2p5_main3",
        "label": "Category profile-only best family (seeds 100/101/102)",
        "patterns": [
            "beauty2013_metadata_v36_source075_w15_catprofileonly_m2p5.json",
            "beauty2013_metadata_v39_profile_seed_m2p5_seed101.json",
            "beauty2013_metadata_v39_profile_seed_m2p5_remaining_seed102.json",
        ],
    },
    {
        "family": "category_profile_m2p5_all",
        "label": "Category profile-only all observed seeds",
        "patterns": [
            "beauty2013_metadata_v36_source075_w15_catprofileonly_m2p5.json",
            "beauty2013_metadata_v39_profile_seed_m2p5_seed101.json",
            "beauty2013_metadata_v39_profile_seed_m2p5_remaining_seed102.json",
            "beauty2013_metadata_v39_profile_seed_m2p5_remaining_seed103.json",
            "beauty2013_metadata_v43_profile_seed104_seed104.json",
            "beauty2013_metadata_v50*_profile_seed*.json",
        ],
    },
]


def _expand_patterns(checkpoint_dir: Path, patterns: list[str]) -> list[Path]:
    matches: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        pattern_matches = sorted(checkpoint_dir.glob(pattern))
        for path in pattern_matches:
            resolved = path.resolve()
            if resolved in seen:
                continue
            matches.append(path)
            seen.add(resolved)
    return matches


def _load_run(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    history = data.get("history") or {}
    best = history.get("best_test_metrics") or {}
    config = data.get("config") or {}
    return {
        "path": str(path),
        "experiment_name": config.get("experiment_name", path.stem),
        "seed": config.get("random_seed"),
        "best_epoch": history.get("best_epoch"),
        "stop_epoch": history.get("stop_epoch"),
        "ndcg10": best.get("NDCG@10"),
        "hr10": best.get("HR@10"),
    }


def _dedupe_best_seed_runs(runs: list[dict]) -> list[dict]:
    best_by_seed: dict[int, dict] = {}
    no_seed_runs: list[dict] = []
    for run in runs:
        seed = run.get("seed")
        ndcg = run.get("ndcg10")
        if seed is None or ndcg is None:
            no_seed_runs.append(run)
            continue
        seed = int(seed)
        existing = best_by_seed.get(seed)
        if existing is None or float(ndcg) > float(existing["ndcg10"]):
            best_by_seed[seed] = run
    deduped = list(best_by_seed.values()) + no_seed_runs
    return sorted(deduped, key=lambda row: (row["seed"] is None, row["seed"], row["experiment_name"]))


def _summarize_family(family: str, label: str, runs: list[dict], baseline_mean: float | None) -> dict:
    ndcgs = [float(run["ndcg10"]) for run in runs if run.get("ndcg10") is not None]
    hrs = [float(run["hr10"]) for run in runs if run.get("hr10") is not None]
    best_ndcg = max(ndcgs) if ndcgs else None
    mean_ndcg = mean(ndcgs) if ndcgs else None
    std_ndcg = pstdev(ndcgs) if len(ndcgs) > 1 else 0.0
    mean_hr = mean(hrs) if hrs else None

    best_relative_gain = None
    mean_relative_gain = None
    if baseline_mean and baseline_mean != 0.0 and best_ndcg is not None and mean_ndcg is not None:
        best_relative_gain = (best_ndcg / baseline_mean - 1.0) * 100.0
        mean_relative_gain = (mean_ndcg / baseline_mean - 1.0) * 100.0

    return {
        "family": family,
        "label": label,
        "seed_count": len(runs),
        "mean_ndcg10": mean_ndcg,
        "std_ndcg10": std_ndcg,
        "best_ndcg10": best_ndcg,
        "mean_hr10": mean_hr,
        "best_relative_gain_vs_baseline_mean_pct": best_relative_gain,
        "mean_relative_gain_vs_baseline_mean_pct": mean_relative_gain,
    }


def _format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "-"
    if math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def _format_pct(value: float | None) -> str:
    if value is None:
        return "-"
    if math.isnan(value):
        return "-"
    return f"{value:.2f}%"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown(path: Path, aggregates: list[dict], detailed_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Beauty2013 Seed Comparison",
        "",
        "## Aggregate Summary",
        "",
        "| Family | Seeds | Mean NDCG@10 | Std NDCG@10 | Best NDCG@10 | Mean HR@10 | Mean Gain vs Baseline Mean | Best Gain vs Baseline Mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregates:
        lines.append(
            "| {label} | {seed_count} | {mean_ndcg10} | {std_ndcg10} | {best_ndcg10} | {mean_hr10} | {mean_gain} | {best_gain} |".format(
                label=row["label"],
                seed_count=row["seed_count"],
                mean_ndcg10=_format_float(row["mean_ndcg10"], 9),
                std_ndcg10=_format_float(row["std_ndcg10"], 9),
                best_ndcg10=_format_float(row["best_ndcg10"], 9),
                mean_hr10=_format_float(row["mean_hr10"], 9),
                mean_gain=_format_pct(row["mean_relative_gain_vs_baseline_mean_pct"]),
                best_gain=_format_pct(row["best_relative_gain_vs_baseline_mean_pct"]),
            )
        )

    lines.extend(
        [
            "",
            "## Per-Run Detail",
            "",
            "| Family | Experiment | Seed | Best Epoch | Stop Epoch | Test NDCG@10 | Test HR@10 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in detailed_rows:
        lines.append(
            "| {family} | {experiment_name} | {seed} | {best_epoch} | {stop_epoch} | {ndcg10} | {hr10} |".format(
                family=row["family"],
                experiment_name=row["experiment_name"],
                seed=row["seed"],
                best_epoch=row["best_epoch"],
                stop_epoch=row["stop_epoch"],
                ndcg10=_format_float(row["ndcg10"], 9),
                hr10=_format_float(row["hr10"], 9),
            )
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def write_latex(path: Path, aggregates: list[dict], detailed_rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Family & Seeds & Mean NDCG@10 & Std NDCG@10 & Best NDCG@10 & Mean Gain (\%) & Best Gain (\%) \\",
        r"\midrule",
    ]
    for row in aggregates:
        mean_gain = "-" if row["mean_relative_gain_vs_baseline_mean_pct"] is None else f'{row["mean_relative_gain_vs_baseline_mean_pct"]:.2f}'
        best_gain = "-" if row["best_relative_gain_vs_baseline_mean_pct"] is None else f'{row["best_relative_gain_vs_baseline_mean_pct"]:.2f}'
        lines.append(
            "{label} & {seed_count} & {mean_ndcg} & {std_ndcg} & {best_ndcg} & {mean_gain} & {best_gain} \\\\".format(
                label=_latex_escape(row["label"]),
                seed_count=row["seed_count"],
                mean_ndcg=_format_float(row["mean_ndcg10"], 6),
                std_ndcg=_format_float(row["std_ndcg10"], 6),
                best_ndcg=_format_float(row["best_ndcg10"], 6),
                mean_gain=mean_gain,
                best_gain=best_gain,
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Family & Experiment & Seed & Best Epoch & Stop Epoch & NDCG@10 \\",
            r"\midrule",
        ]
    )
    for row in detailed_rows:
        lines.append(
            "{family} & {experiment_name} & {seed} & {best_epoch} & {stop_epoch} & {ndcg} \\\\".format(
                family=_latex_escape(row["family"]),
                experiment_name=_latex_escape(row["experiment_name"]),
                seed=row["seed"],
                best_epoch=row["best_epoch"],
                stop_epoch=row["stop_epoch"],
                ndcg=_format_float(row["ndcg10"], 6),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_matched_seed_rows(source_runs: list[dict], category_runs: list[dict]) -> list[dict]:
    def _best_by_seed(runs: list[dict]) -> dict[int, dict]:
        best: dict[int, dict] = {}
        for run in runs:
            seed = run.get("seed")
            ndcg = run.get("ndcg10")
            if seed is None or ndcg is None:
                continue
            existing = best.get(int(seed))
            if existing is None or float(ndcg) > float(existing["ndcg10"]):
                best[int(seed)] = run
        return best

    source_by_seed = _best_by_seed(source_runs)
    category_by_seed = _best_by_seed(category_runs)
    rows = []
    for seed in sorted(set(source_by_seed) & set(category_by_seed)):
        source = source_by_seed[seed]
        category = category_by_seed[seed]
        source_ndcg = float(source["ndcg10"])
        category_ndcg = float(category["ndcg10"])
        delta = category_ndcg - source_ndcg
        relative = 0.0 if source_ndcg == 0.0 else (category_ndcg / source_ndcg - 1.0) * 100.0
        rows.append(
            {
                "seed": seed,
                "source_experiment": source["experiment_name"],
                "source_ndcg10": source_ndcg,
                "source_hr10": source.get("hr10"),
                "category_experiment": category["experiment_name"],
                "category_ndcg10": category_ndcg,
                "category_hr10": category.get("hr10"),
                "delta_ndcg10": delta,
                "relative_gain_vs_source_pct": relative,
            }
        )
    return rows


def write_pairwise_csv(path: Path, rows: list[dict]) -> None:
    write_csv(
        path,
        rows,
        [
            "seed",
            "source_experiment",
            "source_ndcg10",
            "source_hr10",
            "category_experiment",
            "category_ndcg10",
            "category_hr10",
            "delta_ndcg10",
            "relative_gain_vs_source_pct",
        ],
    )


def write_pairwise_markdown(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Beauty2013 Matched-Seed Comparison",
        "",
        "| Seed | Source075 Experiment | Source NDCG@10 | Category-Profile Experiment | Category NDCG@10 | Delta | Relative Gain vs Source |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {seed} | {source_experiment} | {source_ndcg10} | {category_experiment} | {category_ndcg10} | {delta_ndcg10} | {relative_gain} |".format(
                seed=row["seed"],
                source_experiment=row["source_experiment"],
                source_ndcg10=_format_float(row["source_ndcg10"], 9),
                category_experiment=row["category_experiment"],
                category_ndcg10=_format_float(row["category_ndcg10"], 9),
                delta_ndcg10=_format_float(row["delta_ndcg10"], 9),
                relative_gain=_format_pct(row["relative_gain_vs_source_pct"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pairwise_latex(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{rllrrr}",
        r"\toprule",
        r"Seed & Source Experiment & Category Experiment & Source NDCG@10 & Category NDCG@10 & Delta \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{seed} & {source_experiment} & {category_experiment} & {source_ndcg10} & {category_ndcg10} & {delta_ndcg10} \\\\".format(
                seed=row["seed"],
                source_experiment=_latex_escape(row["source_experiment"]),
                category_experiment=_latex_escape(row["category_experiment"]),
                source_ndcg10=_format_float(row["source_ndcg10"], 6),
                category_ndcg10=_format_float(row["category_ndcg10"], 6),
                delta_ndcg10=_format_float(row["delta_ndcg10"], 6),
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    detailed_rows: list[dict] = []
    family_runs: dict[str, list[dict]] = {}
    family_labels: dict[str, str] = {}
    for spec in FAMILY_SPECS:
        family = str(spec["family"])
        label = str(spec["label"])
        runs = []
        for path in _expand_patterns(CHECKPOINT_DIR, list(spec["patterns"])):
            run = _load_run(path)
            run["family"] = family
            runs.append(run)
        deduped_runs = _dedupe_best_seed_runs(runs)
        detailed_rows.extend(deduped_runs)
        family_runs[family] = deduped_runs
        family_labels[family] = label

    baseline_runs = family_runs.get("baseline", [])
    baseline_ndcgs = [float(row["ndcg10"]) for row in baseline_runs if row.get("ndcg10") is not None]
    baseline_mean = mean(baseline_ndcgs) if baseline_ndcgs else None

    aggregate_rows = [
        _summarize_family(family, family_labels[family], runs, baseline_mean)
        for family, runs in family_runs.items()
    ]

    aggregate_rows = sorted(aggregate_rows, key=lambda row: row["family"])
    detailed_rows = sorted(
        detailed_rows,
        key=lambda row: (row["family"], row["seed"] is None, row["seed"], row["experiment_name"]),
    )

    write_csv(
        SUMMARY_DIR / "beauty2013_seed_comparison_aggregate.csv",
        aggregate_rows,
        [
            "family",
            "label",
            "seed_count",
            "mean_ndcg10",
            "std_ndcg10",
            "best_ndcg10",
            "mean_hr10",
            "mean_relative_gain_vs_baseline_mean_pct",
            "best_relative_gain_vs_baseline_mean_pct",
        ],
    )
    write_csv(
        SUMMARY_DIR / "beauty2013_seed_comparison_runs.csv",
        detailed_rows,
        [
            "family",
            "experiment_name",
            "seed",
            "best_epoch",
            "stop_epoch",
            "ndcg10",
            "hr10",
            "path",
        ],
    )
    write_markdown(SUMMARY_DIR / "beauty2013_seed_comparison.md", aggregate_rows, detailed_rows)
    write_latex(SUMMARY_DIR / "beauty2013_seed_comparison.tex", aggregate_rows, detailed_rows)

    matched_rows = build_matched_seed_rows(
        family_runs.get("source075_w15_all", []),
        family_runs.get("category_profile_m2p5_all", []),
    )
    write_pairwise_csv(SUMMARY_DIR / "beauty2013_matched_seed_comparison.csv", matched_rows)
    write_pairwise_markdown(SUMMARY_DIR / "beauty2013_matched_seed_comparison.md", matched_rows)
    write_pairwise_latex(SUMMARY_DIR / "beauty2013_matched_seed_comparison.tex", matched_rows)
    print(
        json.dumps(
            {
                "aggregate_rows": aggregate_rows,
                "detailed_rows": detailed_rows,
                "matched_rows": matched_rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
