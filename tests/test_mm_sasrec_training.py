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


class MultimodalSASRecTrainingTests(unittest.TestCase):
    def test_train_mm_sasrec_smoke_runs_with_category_memory_only(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local Python environment")

        from mmsrec.data.beauty import BeautyDataConfig, build_dataset
        from mmsrec.multimodal.sasrec import MMTrainConfig, train_mm_sasrec

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
                    {"asin": "A", "title": "Alpha", "description": "a", "categories": [["Beauty", "Hair Care"]]},
                    {"asin": "B", "title": "Beta", "description": "b", "categories": [["Beauty", "Skin Care"]]},
                    {"asin": "C", "title": "Gamma", "description": "c", "categories": [["Beauty", "Hair Care"]]},
                    {"asin": "D", "title": "Delta", "description": "d", "categories": [["Beauty", "Fragrance"]]},
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

            content_path = root / "artifacts" / "content_features.pt"
            torch.save(
                {
                    "item_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                    "text_embeddings": torch.randn(4, 4),
                    "image_embeddings": torch.randn(4, 6),
                    "category_embeddings": torch.randn(4, 3),
                    "text_available": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    "image_available": torch.tensor([1, 1, 1, 0], dtype=torch.int64),
                    "category_available": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    "asins": ["A", "B", "C", "D"],
                },
                content_path,
            )

            config = MMTrainConfig(
                root_dir=str(root),
                content_features_path=str(content_path),
                checkpoint_dir=str(root / "checkpoints"),
                experiment_name="mm_sasrec_category_memory_smoke",
                device="cpu",
                epochs=3,
                batch_size=2,
                eval_batch_size=2,
                learning_rate=1e-3,
                hidden_size=8,
                num_heads=2,
                num_blocks=1,
                dropout=0.1,
                topk=(1, 2, 3),
                early_stop_metric="NDCG@1",
                early_stop_patience=1,
                early_stop_min_delta=0.0,
                random_seed=7,
                target_block_dim=4,
                use_category_memory=True,
            )
            result = train_mm_sasrec(config)

            self.assertEqual(result["config"]["experiment_name"], "mm_sasrec_category_memory_smoke")
            self.assertGreaterEqual(len(result["history"]["train_loss"]), 1)
            self.assertGreaterEqual(len(result["history"]["val_metrics"]), 1)
            self.assertTrue(Path(result["checkpoint_path"]).exists())

    def test_train_mm_sasrec_records_plateau_scheduler_learning_rates(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local Python environment")

        from mmsrec.data.beauty import BeautyDataConfig, build_dataset
        from mmsrec.multimodal.sasrec import MMTrainConfig, train_mm_sasrec

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
                    {"asin": "A", "title": "Alpha", "description": "a"},
                    {"asin": "B", "title": "Beta", "description": "b"},
                    {"asin": "C", "title": "Gamma", "description": "c"},
                    {"asin": "D", "title": "Delta", "description": "d"},
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

            content_path = root / "artifacts" / "content_features.pt"
            torch.save(
                {
                    "item_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                    "text_embeddings": torch.randn(4, 4),
                    "image_embeddings": torch.randn(4, 6),
                    "text_available": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    "image_available": torch.tensor([1, 1, 1, 0], dtype=torch.int64),
                    "asins": ["A", "B", "C", "D"],
                },
                content_path,
            )

            config = MMTrainConfig(
                root_dir=str(root),
                content_features_path=str(content_path),
                checkpoint_dir=str(root / "checkpoints"),
                experiment_name="mm_sasrec_plateau_scheduler_smoke",
                device="cpu",
                epochs=4,
                batch_size=2,
                eval_batch_size=2,
                learning_rate=1e-3,
                hidden_size=8,
                num_heads=2,
                num_blocks=1,
                dropout=0.1,
                topk=(1, 2, 3),
                early_stop_metric="NDCG@1",
                early_stop_patience=None,
                early_stop_min_delta=0.0,
                random_seed=7,
                target_block_dim=4,
                separate_sequence_fusion=True,
                lr_scheduler="plateau",
                lr_scheduler_min_lr=1e-4,
                lr_scheduler_factor=0.5,
                lr_scheduler_patience=0,
                lr_scheduler_threshold=1.0,
            )
            result = train_mm_sasrec(config)

            learning_rates = result["history"]["learning_rates"]
            self.assertEqual(len(learning_rates), 4)
            self.assertAlmostEqual(learning_rates[0], 1e-3, places=10)
            self.assertAlmostEqual(learning_rates[1], 1e-3, places=10)
            self.assertAlmostEqual(learning_rates[2], 5e-4, places=10)
            self.assertAlmostEqual(learning_rates[3], 2.5e-4, places=10)

    def test_train_mm_sasrec_records_cosine_scheduler_learning_rates(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local Python environment")

        from mmsrec.data.beauty import BeautyDataConfig, build_dataset
        from mmsrec.multimodal.sasrec import MMTrainConfig, train_mm_sasrec

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
                    {"asin": "A", "title": "Alpha", "description": "a"},
                    {"asin": "B", "title": "Beta", "description": "b"},
                    {"asin": "C", "title": "Gamma", "description": "c"},
                    {"asin": "D", "title": "Delta", "description": "d"},
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

            content_path = root / "artifacts" / "content_features.pt"
            torch.save(
                {
                    "item_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                    "text_embeddings": torch.randn(4, 4),
                    "image_embeddings": torch.randn(4, 6),
                    "text_available": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    "image_available": torch.tensor([1, 1, 1, 0], dtype=torch.int64),
                    "asins": ["A", "B", "C", "D"],
                },
                content_path,
            )

            config = MMTrainConfig(
                root_dir=str(root),
                content_features_path=str(content_path),
                checkpoint_dir=str(root / "checkpoints"),
                experiment_name="mm_sasrec_cosine_scheduler_smoke",
                device="cpu",
                epochs=4,
                batch_size=2,
                eval_batch_size=2,
                learning_rate=1e-3,
                hidden_size=8,
                num_heads=2,
                num_blocks=1,
                dropout=0.1,
                topk=(1, 2, 3),
                early_stop_metric="NDCG@1",
                early_stop_patience=None,
                early_stop_min_delta=0.0,
                random_seed=7,
                target_block_dim=4,
                separate_sequence_fusion=True,
                lr_scheduler="cosine",
                lr_scheduler_min_lr=1e-4,
                lr_scheduler_warmup_epochs=2,
                lr_scheduler_warmup_start_factor=0.2,
            )
            result = train_mm_sasrec(config)

            learning_rates = result["history"]["learning_rates"]
            self.assertEqual(len(learning_rates), 4)
            self.assertAlmostEqual(learning_rates[0], 6e-4, places=10)
            self.assertAlmostEqual(learning_rates[1], 1e-3, places=10)
            self.assertAlmostEqual(learning_rates[2], 1e-3, places=10)
            self.assertAlmostEqual(learning_rates[3], 1e-4, places=10)

    def test_train_mm_sasrec_smoke_runs_with_external_content_features(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local Python environment")

        from mmsrec.data.beauty import BeautyDataConfig, build_dataset
        from mmsrec.multimodal.sasrec import MMTrainConfig, train_mm_sasrec

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
                    {"asin": "A", "title": "Alpha", "description": "a"},
                    {"asin": "B", "title": "Beta", "description": "b"},
                    {"asin": "C", "title": "Gamma", "description": "c"},
                    {"asin": "D", "title": "Delta", "description": "d"},
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

            content_path = root / "artifacts" / "content_features.pt"
            torch.save(
                {
                    "item_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                    "text_embeddings": torch.randn(4, 4),
                    "image_embeddings": torch.randn(4, 6),
                    "text_available": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    "image_available": torch.tensor([1, 1, 1, 0], dtype=torch.int64),
                    "asins": ["A", "B", "C", "D"],
                },
                content_path,
            )

            config = MMTrainConfig(
                root_dir=str(root),
                content_features_path=str(content_path),
                checkpoint_dir=str(root / "checkpoints"),
                experiment_name="mm_sasrec_smoke",
                device="cpu",
                epochs=3,
                batch_size=2,
                eval_batch_size=2,
                learning_rate=1e-3,
                hidden_size=8,
                num_heads=2,
                num_blocks=1,
                dropout=0.1,
                topk=(1, 2, 3),
                early_stop_metric="NDCG@1",
                early_stop_patience=1,
                early_stop_min_delta=0.0,
                random_seed=7,
                target_block_dim=4,
                separate_sequence_fusion=True,
            )
            result = train_mm_sasrec(config)

            self.assertEqual(result["config"]["root_dir"], str(root))
            self.assertGreaterEqual(len(result["history"]["train_loss"]), 1)
            self.assertGreaterEqual(len(result["history"]["val_metrics"]), 1)
            self.assertIn("best_val_metrics", result["history"])
            self.assertIn("best_test_metrics", result["history"])
            self.assertTrue(Path(result["checkpoint_path"]).exists())


if __name__ == "__main__":
    unittest.main()
