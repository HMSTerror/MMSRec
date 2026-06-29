import tempfile
import unittest
from pathlib import Path


def _write_checkpoint(path: Path, *, experiment_name: str, seed: int, ndcg10: float, hr10: float = 0.08) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "{\n"
            '  "history": {\n'
            '    "best_epoch": 4,\n'
            '    "stop_epoch": 9,\n'
            '    "best_test_metrics": {\n'
            f'      "NDCG@10": {ndcg10},\n'
            f'      "HR@10": {hr10}\n'
            "    }\n"
            "  },\n"
            '  "config": {\n'
            f'    "experiment_name": "{experiment_name}",\n'
            f'    "random_seed": {seed}\n'
            "  }\n"
            "}\n"
        ),
        encoding="utf-8",
    )


def _sample_context() -> dict:
    return {
        "headline": {
            "baseline_experiment": "beauty2013_paper_v1_baseline",
            "baseline_ndcg10": 0.04442677095272959,
            "best_experiment": "beauty2013_metadata_v39_profile_seed_m2p5_remaining_seed102",
            "best_ndcg10": 0.05379175070654626,
            "relative_gain_vs_baseline": 0.21079586818904938,
            "baseline_mean_ndcg10": 0.04543978570341635,
            "best_relative_gain_vs_baseline_mean": 0.183802913544594,
        },
        "matched_robustness": {
            "seed_count": 10,
            "wins": 7,
            "losses": 3,
            "ties": 0,
            "mean_source_ndcg10": 0.0512101941747265,
            "mean_category_ndcg10": 0.05191262781170142,
            "mean_delta_ndcg10": 0.000702433636974921,
            "median_delta_ndcg10": 0.00025929110404413966,
            "std_source_ndcg10": 0.0011195765261739505,
            "std_category_ndcg10": 0.0010127844942286871,
            "std_delta_ndcg10": 0.001198680251231151,
            "mean_source_relative_gain_vs_baseline_mean": 0.12699022193839937,
            "mean_category_relative_gain_vs_baseline_mean": 0.1424487815706934,
            "mean_delta_bootstrap_ci": [-0.00003144090939719499, 0.001458498598191279],
            "source_mean_bootstrap_ci": [0.05051854324954591, 0.05189825436585567],
            "category_mean_bootstrap_ci": [0.051306736360360206, 0.0525599284608132],
            "sign_test": {
                "effective_pairs": 10,
                "wins": 7,
                "losses": 3,
                "one_sided_p_value": 0.171875,
                "two_sided_p_value": 0.34375,
            },
        },
        "main_results": [
            {
                "model": "PyTorch SASRec baseline",
                "seed_policy": "seed100",
                "seed_label": "100",
                "ndcg10": 0.04442677095272959,
                "hr10": 0.0685954478379466,
                "gain_pct": 0.0,
            },
            {
                "model": "Best source-text multimodal family (source075_w15)",
                "seed_policy": "seed100",
                "seed_label": "100",
                "ndcg10": 0.05309571530702678,
                "hr10": 0.08746590350131914,
                "gain_pct": 19.51,
            },
            {
                "model": "Best category-profile multimodal run",
                "seed_policy": "seed102",
                "seed_label": "102",
                "ndcg10": 0.05379175070654626,
                "hr10": 0.09028305683495058,
                "gain_pct": 21.08,
            },
        ],
        "search_trajectory": [
            {
                "variant": "SASRec baseline",
                "ndcg10": 0.04442677095272959,
                "gain_pct": 0.0,
                "role": "reference point",
            },
            {
                "variant": "Best category-profile seed hit (v39 seed102)",
                "ndcg10": 0.05379175070654626,
                "gain_pct": 21.08,
                "role": "strongest headline result",
            },
        ],
        "negative_followups": [
            {
                "follow_up": "Text-profile blend (v45)",
                "best_outcome": "best 0.053300, still below v36",
                "conclusion": "Reintroducing weak text-profile scoring does not help.",
            },
            {
                "follow_up": "Plateau seed rescue (v47)",
                "best_outcome": "weak seeds remain at 0.051038 / 0.051003",
                "conclusion": "Schedule changes do not solve robustness.",
            },
            {
                "follow_up": "Weak-seed gate sweep (v48)",
                "best_outcome": "nearby gates do not beat original weak seeds",
                "conclusion": "Local gate mismatch is not the main issue.",
            },
        ],
    }


class Beauty2013SubmissionBundleTests(unittest.TestCase):
    def test_build_search_trajectory_uses_paper_v3_for_the_early_multimodal_milestone(self) -> None:
        from experiments import beauty2013_submission_bundle as bundle

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            _write_checkpoint(
                checkpoint_dir / "beauty2013_paper_v1_baseline.json",
                experiment_name="beauty2013_paper_v1_baseline",
                seed=100,
                ndcg10=0.04442677095272959,
            )
            _write_checkpoint(
                checkpoint_dir / "beauty2013_paper_v3_mm_decomp_full_text_biased_tuned.json",
                experiment_name="beauty2013_paper_v3_mm_decomp_full_text_biased_tuned",
                seed=100,
                ndcg10=0.050209541466039335,
            )
            _write_checkpoint(
                checkpoint_dir / "beauty2013_metadata_v3_df1_full_tuned.json",
                experiment_name="beauty2013_metadata_v3_df1_full_tuned",
                seed=100,
                ndcg10=0.05162438570833284,
            )
            _write_checkpoint(
                checkpoint_dir / "beauty2013_metadata_v5_full_tuned.json",
                experiment_name="beauty2013_metadata_v5_full_tuned",
                seed=100,
                ndcg10=0.052411247033290295,
            )
            _write_checkpoint(
                checkpoint_dir / "beauty2013_metadata_v8_w15_full_tuned.json",
                experiment_name="beauty2013_metadata_v8_w15_full_tuned",
                seed=100,
                ndcg10=0.052646413040628386,
            )
            _write_checkpoint(
                checkpoint_dir / "beauty2013_metadata_v16_source075_w15_full_tuned.json",
                experiment_name="beauty2013_metadata_v16_source075_w15_full_tuned",
                seed=100,
                ndcg10=0.05309571530702678,
            )
            _write_checkpoint(
                checkpoint_dir / "beauty2013_metadata_v36_source075_w15_catprofileonly_m2p5.json",
                experiment_name="beauty2013_metadata_v36_source075_w15_catprofileonly_m2p5",
                seed=100,
                ndcg10=0.053309771659891945,
            )
            _write_checkpoint(
                checkpoint_dir / "beauty2013_metadata_v39_profile_seed_m2p5_remaining_seed102.json",
                experiment_name="beauty2013_metadata_v39_profile_seed_m2p5_remaining_seed102",
                seed=102,
                ndcg10=0.05379175070654626,
            )

            rows = bundle._build_search_trajectory(checkpoint_dir, 0.04442677095272959)

            self.assertEqual(rows[1]["variant"], "Early multimodal decomposition (v3)")
            self.assertAlmostEqual(rows[1]["ndcg10"], 0.050209541466039335, places=12)
            self.assertEqual(rows[2]["variant"], "Metadata + source text full tuned")
            self.assertAlmostEqual(rows[2]["ndcg10"], 0.05162438570833284, places=12)

    def test_write_submission_bundle_creates_paper_ready_reports(self) -> None:
        from experiments import beauty2013_submission_bundle as bundle

        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir) / "reports"
            bundle.write_submission_bundle(_sample_context(), report_dir=report_dir)

            tables_md = (report_dir / "beauty2013_submission_tables.md").read_text(encoding="utf-8")
            tables_tex = (report_dir / "beauty2013_submission_tables.tex").read_text(encoding="utf-8")
            notes_md = (report_dir / "beauty2013_b_conference_submission_notes.md").read_text(encoding="utf-8")
            draft_md = (report_dir / "beauty2013_experimental_section_draft.md").read_text(encoding="utf-8")

            self.assertIn(
                "| Best category-profile multimodal run | seed102 | 0.053791751 | 0.090283057 | 21.08% |",
                tables_md,
            )
            self.assertIn("Best category-profile run & 102 & 0.053792 & 0.090283 & 21.08", tables_tex)
            self.assertIn("relative gain vs baseline: `+21.08%`", notes_md)
            self.assertIn("category-profile wins `7 / 10` matched seeds", draft_md)


if __name__ == "__main__":
    unittest.main()
