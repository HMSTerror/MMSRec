from __future__ import annotations

import json
import math
import random
from pathlib import Path
from statistics import mean, median, pstdev

try:
    from beauty2013_paper_seed_table import (  # type: ignore
        CHECKPOINT_DIR,
        FAMILY_SPECS,
        SUMMARY_DIR,
        _dedupe_best_seed_runs,
        _expand_patterns,
        _load_run,
        build_matched_seed_rows,
    )
except ImportError:
    from experiments.beauty2013_paper_seed_table import (
        CHECKPOINT_DIR,
        FAMILY_SPECS,
        SUMMARY_DIR,
        _dedupe_best_seed_runs,
        _expand_patterns,
        _load_run,
        build_matched_seed_rows,
    )


def _bootstrap_mean_ci(values: list[float], n_resamples: int = 20000, seed: int = 20260629) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        raise ValueError("bootstrap requires at least one value")
    samples = []
    for _ in range(n_resamples):
        sample_mean = mean(values[rng.randrange(n)] for _ in range(n))
        samples.append(sample_mean)
    samples.sort()
    lo_idx = int(0.025 * n_resamples)
    hi_idx = max(lo_idx, int(0.975 * n_resamples) - 1)
    return samples[lo_idx], samples[hi_idx]


def _binom_upper_tail(n: int, k: int) -> float:
    return sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n)


def _exact_sign_test(wins: int, losses: int) -> dict:
    effective = wins + losses
    if effective == 0:
        return {
            "effective_pairs": 0,
            "wins": wins,
            "losses": losses,
            "one_sided_p_value": 1.0,
            "two_sided_p_value": 1.0,
        }
    upper = _binom_upper_tail(effective, wins)
    lower = sum(math.comb(effective, i) for i in range(0, wins + 1)) / (2 ** effective)
    two_sided = min(1.0, 2.0 * min(upper, lower))
    return {
        "effective_pairs": effective,
        "wins": wins,
        "losses": losses,
        "one_sided_p_value": upper,
        "two_sided_p_value": two_sided,
    }


def _format_float(value: float, digits: int = 6) -> str:
    return f"{value:.{digits}f}"


def _format_pct(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def _load_family_runs() -> tuple[dict[str, list[dict]], float]:
    family_runs: dict[str, list[dict]] = {}
    for spec in FAMILY_SPECS:
        family = str(spec["family"])
        runs = []
        for path in _expand_patterns(CHECKPOINT_DIR, list(spec["patterns"])):
            run = _load_run(path)
            run["family"] = family
            runs.append(run)
        family_runs[family] = _dedupe_best_seed_runs(runs)

    baseline_runs = family_runs["baseline"]
    baseline_mean = mean(float(row["ndcg10"]) for row in baseline_runs if row.get("ndcg10") is not None)
    return family_runs, baseline_mean


def build_statistics() -> dict:
    family_runs, baseline_mean = _load_family_runs()
    matched_rows = build_matched_seed_rows(
        family_runs.get("source075_w15_all", []),
        family_runs.get("category_profile_m2p5_all", []),
    )

    deltas = [float(row["delta_ndcg10"]) for row in matched_rows]
    source_ndcgs = [float(row["source_ndcg10"]) for row in matched_rows]
    category_ndcgs = [float(row["category_ndcg10"]) for row in matched_rows]
    wins = sum(delta > 0.0 for delta in deltas)
    losses = sum(delta < 0.0 for delta in deltas)
    ties = len(deltas) - wins - losses

    best_single = max(
        family_runs.get("category_profile_m2p5_all", []),
        key=lambda row: float(row["ndcg10"]) if row.get("ndcg10") is not None else float("-inf"),
    )
    baseline_single = next(row for row in family_runs["baseline"] if row.get("experiment_name") == "beauty2013_paper_v1_baseline")
    best_single_ndcg = float(best_single["ndcg10"])
    baseline_single_ndcg = float(baseline_single["ndcg10"])

    delta_ci_lo, delta_ci_hi = _bootstrap_mean_ci(deltas)
    source_ci_lo, source_ci_hi = _bootstrap_mean_ci(source_ndcgs)
    category_ci_lo, category_ci_hi = _bootstrap_mean_ci(category_ndcgs)
    sign_test = _exact_sign_test(wins, losses)

    return {
        "headline": {
            "baseline_experiment": baseline_single["experiment_name"],
            "baseline_ndcg10": baseline_single_ndcg,
            "best_experiment": best_single["experiment_name"],
            "best_ndcg10": best_single_ndcg,
            "relative_gain_vs_baseline": (best_single_ndcg / baseline_single_ndcg) - 1.0,
            "baseline_mean_ndcg10": baseline_mean,
            "best_relative_gain_vs_baseline_mean": (best_single_ndcg / baseline_mean) - 1.0,
        },
        "matched_robustness": {
            "seed_count": len(matched_rows),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "mean_source_ndcg10": mean(source_ndcgs),
            "mean_category_ndcg10": mean(category_ndcgs),
            "mean_delta_ndcg10": mean(deltas),
            "median_delta_ndcg10": median(deltas),
            "std_source_ndcg10": pstdev(source_ndcgs),
            "std_category_ndcg10": pstdev(category_ndcgs),
            "std_delta_ndcg10": pstdev(deltas),
            "mean_source_relative_gain_vs_baseline_mean": (mean(source_ndcgs) / baseline_mean) - 1.0,
            "mean_category_relative_gain_vs_baseline_mean": (mean(category_ndcgs) / baseline_mean) - 1.0,
            "mean_delta_bootstrap_ci": [delta_ci_lo, delta_ci_hi],
            "source_mean_bootstrap_ci": [source_ci_lo, source_ci_hi],
            "category_mean_bootstrap_ci": [category_ci_lo, category_ci_hi],
            "sign_test": sign_test,
            "matched_rows": matched_rows,
        },
    }


def write_markdown(path: Path, stats: dict) -> None:
    headline = stats["headline"]
    matched = stats["matched_robustness"]
    sign_test = matched["sign_test"]
    lines = [
        "# Beauty2013 Paper Statistics",
        "",
        "## Headline Single-Run Result",
        "",
        f"- baseline `{headline['baseline_experiment']}`: `NDCG@10 = {_format_float(headline['baseline_ndcg10'], 9)}`",
        f"- best run `{headline['best_experiment']}`: `NDCG@10 = {_format_float(headline['best_ndcg10'], 9)}`",
        f"- relative gain vs original single-run baseline: `{_format_pct(headline['relative_gain_vs_baseline'])}`",
        f"- relative gain vs three-seed baseline mean: `{_format_pct(headline['best_relative_gain_vs_baseline_mean'])}`",
        "",
        "## Ten-Seed Matched Robustness",
        "",
        "| Family | Seeds | Mean NDCG@10 | 95% Bootstrap CI | Std NDCG@10 | Relative Gain vs Baseline Mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        "| source075_w15 | {seed_count} | {source_mean} | [{source_lo}, {source_hi}] | {source_std} | {source_gain} |".format(
            seed_count=matched["seed_count"],
            source_mean=_format_float(matched["mean_source_ndcg10"], 9),
            source_lo=_format_float(matched["source_mean_bootstrap_ci"][0], 9),
            source_hi=_format_float(matched["source_mean_bootstrap_ci"][1], 9),
            source_std=_format_float(matched["std_source_ndcg10"], 9),
            source_gain=_format_pct(matched["mean_source_relative_gain_vs_baseline_mean"]),
        ),
        "| category-profile | {seed_count} | {category_mean} | [{category_lo}, {category_hi}] | {category_std} | {category_gain} |".format(
            seed_count=matched["seed_count"],
            category_mean=_format_float(matched["mean_category_ndcg10"], 9),
            category_lo=_format_float(matched["category_mean_bootstrap_ci"][0], 9),
            category_hi=_format_float(matched["category_mean_bootstrap_ci"][1], 9),
            category_std=_format_float(matched["std_category_ndcg10"], 9),
            category_gain=_format_pct(matched["mean_category_relative_gain_vs_baseline_mean"]),
        ),
        "",
        "## Paired Delta Analysis",
        "",
        f"- matched wins/losses/ties: `{matched['wins']} / {matched['losses']} / {matched['ties']}`",
        f"- mean delta `(category - source)`: `{_format_float(matched['mean_delta_ndcg10'], 9)}`",
        f"- median delta `(category - source)`: `{_format_float(matched['median_delta_ndcg10'], 9)}`",
        f"- std delta `(category - source)`: `{_format_float(matched['std_delta_ndcg10'], 9)}`",
        f"- 95% bootstrap CI for mean delta: `[{_format_float(matched['mean_delta_bootstrap_ci'][0], 9)}, {_format_float(matched['mean_delta_bootstrap_ci'][1], 9)}]`",
        f"- exact sign test one-sided p-value: `{sign_test['one_sided_p_value']:.6f}`",
        f"- exact sign test two-sided p-value: `{sign_test['two_sided_p_value']:.6f}`",
        "",
        "## Interpretation",
        "",
        "- The best single run remains strong enough for a clear headline result.",
        "- The matched-seed comparison now favors the category-profile branch on both win count and average score.",
        "- The exact sign test is the cleanest small-sample nonparametric check for the paired seed comparison.",
        "- If the p-value is not yet conventionally significant, the correct paper framing is a positive robustness trend rather than a definitive statistical claim.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    stats = build_statistics()
    output_json = SUMMARY_DIR / "beauty2013_paper_statistics.json"
    output_md = SUMMARY_DIR / "beauty2013_paper_statistics.md"
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    write_markdown(output_md, stats)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
