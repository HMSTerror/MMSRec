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


class BeautyPipelineTests(unittest.TestCase):
    def test_build_dataset_creates_expected_artifacts_from_raw_gzip(self) -> None:
        from mmsrec.data.beauty import BeautyDataConfig, build_dataset

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
                ],
            )
            _write_jsonl_gz(
                raw / "meta_Beauty.json.gz",
                [
                    {"asin": "A", "title": "Alpha", "description": "A desc", "categories": [["Beauty", "A"]], "brand": "BrandA", "imUrl": "http://a"},
                    {"asin": "B", "title": "Beta", "description": "B desc", "categories": [["Beauty", "B"]], "brand": "BrandB", "imUrl": "http://b"},
                    {"asin": "C", "title": "Gamma", "description": "C desc", "categories": [["Beauty", "C"]], "brand": "BrandC", "imUrl": "http://c"},
                    {"asin": "D", "title": "Delta", "description": "D desc", "categories": [["Beauty", "D"]], "brand": "BrandD", "imUrl": "http://d"},
                ],
            )

            cfg = BeautyDataConfig(
                root_dir=str(root),
                max_seq_len=5,
                min_user_interactions=2,
                min_item_interactions=2,
            )
            summary = build_dataset(cfg)

            self.assertEqual(summary["user_count"], 2)
            self.assertEqual(summary["item_count"], 4)
            self.assertEqual(summary["interaction_count"], 8)
            self.assertEqual(summary["train_rows"], 2)
            self.assertEqual(summary["val_rows"], 2)
            self.assertEqual(summary["test_rows"], 2)

            artifacts = root / "artifacts"
            self.assertTrue((artifacts / "items.df").exists())
            self.assertTrue((artifacts / "interactions.df").exists())
            self.assertTrue((artifacts / "train_data.df").exists())
            self.assertTrue((artifacts / "val_data.df").exists())
            self.assertTrue((artifacts / "test_data.df").exists())
            self.assertTrue((artifacts / "data_statis.df").exists())
            self.assertTrue((artifacts / "dataset_summary.json").exists())

    def test_export_original_interactions_txt_writes_original_sasrec_format(self) -> None:
        from mmsrec.data.beauty import (
            BeautyDataConfig,
            build_dataset,
            export_original_interactions_txt,
        )

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

            cfg = BeautyDataConfig(
                root_dir=str(root),
                max_seq_len=5,
                min_user_interactions=2,
                min_item_interactions=2,
            )
            build_dataset(cfg)

            output_path = root / "exports" / "Beauty.txt"
            manifest = export_original_interactions_txt(str(root), str(output_path))

            self.assertEqual(manifest["user_count"], 2)
            self.assertEqual(manifest["item_count"], 4)
            self.assertEqual(manifest["interaction_count"], 8)
            self.assertEqual(
                output_path.read_text(encoding="utf-8").splitlines(),
                [
                    "1 1",
                    "1 2",
                    "1 3",
                    "1 4",
                    "2 1",
                    "2 2",
                    "2 3",
                    "2 4",
                ],
            )


if __name__ == "__main__":
    unittest.main()
