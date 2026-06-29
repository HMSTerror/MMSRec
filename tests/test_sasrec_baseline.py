import gzip
import json
import tempfile
import unittest
from pathlib import Path


def _write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


class SASRecBaselineTrainingTests(unittest.TestCase):
    def test_train_sasrec_baseline_smoke_runs_with_early_stopping_outputs(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed in the local Python environment")

        from mmsrec.data.beauty import BeautyDataConfig, build_dataset
        from mmsrec.baselines.sasrec import BaselineTrainConfig, train_sasrec_baseline

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "beauty2013"
            raw = root / "raw"
            _write_jsonl_gz(
                raw / "reviews_Beauty.json.gz",
                [
                    {"reviewerID": "U1", "asin": "A", "unixReviewTime": 1, "overall": 5.0},
                    {"reviewerID": "U1", "asin": "B", "unixReviewTime": 2, "overall": 5.0},
                    {"reviewerID": "U1", "asin": "C", "unixReviewTime": 3, "overall": 5.0},
                    {"reviewerID": "U1", "asin": "D", "unixReviewTime": 4, "overall": 5.0},
                    {"reviewerID": "U2", "asin": "A", "unixReviewTime": 1, "overall": 4.0},
                    {"reviewerID": "U2", "asin": "B", "unixReviewTime": 2, "overall": 4.0},
                    {"reviewerID": "U2", "asin": "C", "unixReviewTime": 3, "overall": 4.0},
                    {"reviewerID": "U2", "asin": "D", "unixReviewTime": 4, "overall": 4.0},
                    {"reviewerID": "U3", "asin": "A", "unixReviewTime": 1, "overall": 4.0},
                    {"reviewerID": "U3", "asin": "B", "unixReviewTime": 2, "overall": 4.0},
                    {"reviewerID": "U3", "asin": "C", "unixReviewTime": 3, "overall": 4.0},
                    {"reviewerID": "U3", "asin": "D", "unixReviewTime": 4, "overall": 4.0},
                ],
            )
            _write_jsonl_gz(
                raw / "meta_Beauty.json.gz",
                [
                    {"asin": "A", "title": "Alpha"},
                    {"asin": "B", "title": "Beta"},
                    {"asin": "C", "title": "Gamma"},
                    {"asin": "D", "title": "Delta"},
                ],
            )

            build_dataset(
                BeautyDataConfig(
                    root_dir=str(root),
                    max_seq_len=5,
                    min_user_interactions=2,
                    min_item_interactions=2,
                )
            )

            config = BaselineTrainConfig(
                root_dir=str(root),
                checkpoint_dir=str(root / "checkpoints"),
                experiment_name="sasrec_smoke",
                device="cpu",
                epochs=3,
                batch_size=2,
                eval_batch_size=2,
                learning_rate=1e-3,
                hidden_size=8,
                num_heads=1,
                num_blocks=1,
                dropout=0.1,
                topk=(1, 2, 3),
                early_stop_metric="NDCG@1",
                early_stop_patience=1,
                early_stop_min_delta=0.0,
                random_seed=7,
            )
            result = train_sasrec_baseline(config)

            self.assertEqual(result["config"]["root_dir"], str(root))
            self.assertGreaterEqual(len(result["history"]["train_loss"]), 1)
            self.assertGreaterEqual(len(result["history"]["val_metrics"]), 1)
            self.assertIn("best_epoch", result["history"])
            self.assertIn("best_metric_name", result["history"])
            self.assertIn("best_val_metrics", result["history"])
            self.assertIn("best_test_metrics", result["history"])
            self.assertTrue(Path(result["checkpoint_path"]).exists())


if __name__ == "__main__":
    unittest.main()
