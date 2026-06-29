import tempfile
import unittest
from pathlib import Path


class MetadataContentFeatureTests(unittest.TestCase):
    def test_compose_weighted_text_inputs_can_repeat_structured_fields(self) -> None:
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas is not installed in the local Python environment")

        from mmsrec.features.metadata import _compose_weighted_text_inputs

        items = pd.DataFrame(
            [
                {
                    "title": "Repair Shampoo",
                    "categories": "Beauty Hair Care",
                    "brand": "BrandOne",
                    "text_input": "Repair Shampoo deep repair formula Beauty Hair Care BrandOne",
                }
            ]
        )

        texts = _compose_weighted_text_inputs(
            items,
            extra_title_repeats=2,
            extra_category_repeats=1,
            extra_brand_repeats=1,
        )

        self.assertEqual(
            texts[0],
            "Repair Shampoo Repair Shampoo Beauty Hair Care BrandOne "
            "Repair Shampoo deep repair formula Beauty Hair Care BrandOne",
        )

    def test_build_metadata_content_features_can_append_and_scale_title_embeddings(self) -> None:
        try:
            import pandas as pd
            import torch
        except ImportError:
            self.skipTest("pandas/torch are not installed in the local Python environment")

        from mmsrec.features.metadata import build_metadata_content_features

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            items_path = root / "items.df"
            source_path = root / "source_content_features.pt"
            base_output = root / "base.pt"
            scaled_output = root / "scaled.pt"

            pd.DataFrame(
                [
                    {"item_id": 0, "asin": "A", "title": "Repair Shampoo", "text_input": "repair shampoo dry hair formula"},
                    {"item_id": 1, "asin": "B", "title": "Vitamin Serum", "text_input": "vitamin serum bright skin daily use"},
                    {"item_id": 2, "asin": "C", "title": "Repair Shampoo", "text_input": "repair shampoo damaged hair care"},
                    {"item_id": 3, "asin": "D", "title": "Matte Lipstick", "text_input": "matte lipstick long wear red"},
                ]
            ).to_pickle(items_path)

            torch.save(
                {
                    "item_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                    "text_embeddings": torch.randn(4, 2),
                    "image_embeddings": torch.randn(4, 3),
                    "text_available": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    "image_available": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    "asins": ["A", "B", "C", "D"],
                },
                source_path,
            )

            common_kwargs = dict(
                items_df_path=str(items_path),
                source_content_features_path=str(source_path),
                text_svd_dim=3,
                max_text_features=16,
                ngram_max=1,
                append_title_embeddings=True,
                title_svd_dim=2,
            )
            build_metadata_content_features(output_path=str(base_output), **common_kwargs)
            build_metadata_content_features(
                output_path=str(scaled_output),
                title_weight=2.5,
                **common_kwargs,
            )

            base_payload = torch.load(base_output, map_location="cpu")
            scaled_payload = torch.load(scaled_output, map_location="cpu")
            self.assertEqual(tuple(base_payload["text_embeddings"].shape), (4, 5))
            self.assertTrue(torch.allclose(base_payload["text_embeddings"][0, -2:], base_payload["text_embeddings"][2, -2:], atol=1e-5))
            self.assertTrue(torch.allclose(scaled_payload["text_embeddings"][:, -2:], base_payload["text_embeddings"][:, -2:] * 2.5, atol=1e-5))

    def test_build_metadata_content_features_aligns_images_and_builds_text_embeddings(self) -> None:
        try:
            import pandas as pd
            import torch
        except ImportError:
            self.skipTest("pandas/torch are not installed in the local Python environment")

        try:
            from mmsrec.features.metadata import build_metadata_content_features
        except ImportError as exc:
            self.fail(f"metadata feature builder import failed: {exc}")

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            items_path = root / "items.df"
            source_path = root / "source_content_features.pt"
            output_path = root / "metadata_content_features.pt"

            pd.DataFrame(
                [
                    {"item_id": 0, "asin": "B", "text_input": "beta skincare serum vitamin c"},
                    {"item_id": 1, "asin": "A", "text_input": "alpha hair shampoo repair"},
                    {"item_id": 2, "asin": "C", "text_input": "gamma makeup palette eyeshadow"},
                    {"item_id": 3, "asin": "D", "text_input": "delta perfume floral women"},
                ]
            ).to_pickle(items_path)

            torch.save(
                {
                    "item_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                    "text_embeddings": torch.randn(4, 5),
                    "image_embeddings": torch.tensor(
                        [
                            [10.0, 11.0, 12.0],
                            [20.0, 21.0, 22.0],
                            [30.0, 31.0, 32.0],
                            [40.0, 41.0, 42.0],
                        ],
                        dtype=torch.float32,
                    ),
                    "text_available": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    "image_available": torch.tensor([1, 1, 0, 1], dtype=torch.int64),
                    "asins": ["A", "B", "C", "D"],
                },
                source_path,
            )

            summary = build_metadata_content_features(
                items_df_path=str(items_path),
                source_content_features_path=str(source_path),
                output_path=str(output_path),
                text_svd_dim=3,
                max_text_features=16,
                ngram_max=1,
            )

            self.assertEqual(summary["item_count"], 4)
            self.assertEqual(summary["text_embedding_dim"], 3)
            self.assertEqual(summary["image_embedding_dim"], 3)
            self.assertTrue(output_path.exists())

            payload = torch.load(output_path, map_location="cpu")
            self.assertEqual(tuple(payload["text_embeddings"].shape), (4, 3))
            self.assertEqual(tuple(payload["image_embeddings"].shape), (4, 3))
            self.assertEqual(payload["asins"], ["B", "A", "C", "D"])
            self.assertEqual(int(payload["text_available"].sum().item()), 4)
            self.assertEqual(int(payload["image_available"].sum().item()), 3)
            self.assertTrue(torch.equal(payload["image_embeddings"][0], torch.tensor([20.0, 21.0, 22.0])))
            self.assertTrue(torch.equal(payload["image_embeddings"][1], torch.tensor([10.0, 11.0, 12.0])))

    def test_build_metadata_content_features_can_append_source_text_embeddings(self) -> None:
        try:
            import pandas as pd
            import torch
        except ImportError:
            self.skipTest("pandas/torch are not installed in the local Python environment")

        from mmsrec.features.metadata import build_metadata_content_features

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            items_path = root / "items.df"
            source_path = root / "source_content_features.pt"
            output_path = root / "metadata_plus_source.pt"

            pd.DataFrame(
                [
                    {"item_id": 0, "asin": "A", "text_input": "alpha shampoo repair"},
                    {"item_id": 1, "asin": "B", "text_input": "beta serum vitamin c"},
                    {"item_id": 2, "asin": "C", "text_input": "gamma perfume floral"},
                    {"item_id": 3, "asin": "D", "text_input": "delta lipstick matte"},
                ]
            ).to_pickle(items_path)

            torch.save(
                {
                    "item_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                    "text_embeddings": torch.tensor(
                        [
                            [1.0, 2.0],
                            [3.0, 4.0],
                            [5.0, 6.0],
                            [7.0, 8.0],
                        ],
                        dtype=torch.float32,
                    ),
                    "image_embeddings": torch.randn(4, 3),
                    "text_available": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    "image_available": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    "asins": ["A", "B", "C", "D"],
                },
                source_path,
            )

            summary = build_metadata_content_features(
                items_df_path=str(items_path),
                source_content_features_path=str(source_path),
                output_path=str(output_path),
                text_svd_dim=3,
                max_text_features=16,
                ngram_max=1,
                append_source_text_embeddings=True,
            )

            payload = torch.load(output_path, map_location="cpu")
            self.assertEqual(summary["text_embedding_dim"], 5)
            self.assertEqual(tuple(payload["text_embeddings"].shape), (4, 5))
            self.assertTrue(torch.equal(payload["text_embeddings"][0, -2:], torch.tensor([1.0, 2.0])))

    def test_build_metadata_content_features_can_append_brand_and_category_embeddings(self) -> None:
        try:
            import pandas as pd
            import torch
        except ImportError:
            self.skipTest("pandas/torch are not installed in the local Python environment")

        from mmsrec.features.metadata import build_metadata_content_features

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            items_path = root / "items.df"
            source_path = root / "source_content_features.pt"
            output_path = root / "metadata_plus_brand_category.pt"

            pd.DataFrame(
                [
                    {"item_id": 0, "asin": "A", "text_input": "alpha shampoo repair", "brand": "BrandOne", "categories": "Beauty Hair Care"},
                    {"item_id": 1, "asin": "B", "text_input": "beta serum vitamin c", "brand": "BrandTwo", "categories": "Beauty Skin Care"},
                    {"item_id": 2, "asin": "C", "text_input": "gamma shampoo nourish", "brand": "BrandOne", "categories": "Beauty Hair Care"},
                    {"item_id": 3, "asin": "D", "text_input": "delta lipstick matte", "brand": "BrandThree", "categories": "Beauty Makeup Lips"},
                ]
            ).to_pickle(items_path)

            torch.save(
                {
                    "item_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                    "text_embeddings": torch.randn(4, 2),
                    "image_embeddings": torch.randn(4, 3),
                    "text_available": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    "image_available": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    "asins": ["A", "B", "C", "D"],
                },
                source_path,
            )

            summary = build_metadata_content_features(
                items_df_path=str(items_path),
                source_content_features_path=str(source_path),
                output_path=str(output_path),
                text_svd_dim=3,
                max_text_features=16,
                ngram_max=1,
                append_brand_embeddings=True,
                append_category_embeddings=True,
                brand_svd_dim=1,
                category_svd_dim=2,
            )

            payload = torch.load(output_path, map_location="cpu")
            self.assertEqual(summary["text_embedding_dim"], 6)
            self.assertEqual(tuple(payload["text_embeddings"].shape), (4, 6))
            self.assertTrue(torch.allclose(payload["text_embeddings"][0, -3:], payload["text_embeddings"][2, -3:], atol=1e-5))

    def test_build_metadata_content_features_can_scale_structured_metadata_blocks(self) -> None:
        try:
            import pandas as pd
            import torch
        except ImportError:
            self.skipTest("pandas/torch are not installed in the local Python environment")

        from mmsrec.features.metadata import build_metadata_content_features

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            items_path = root / "items.df"
            source_path = root / "source_content_features.pt"
            base_output = root / "base.pt"
            scaled_output = root / "scaled.pt"

            pd.DataFrame(
                [
                    {"item_id": 0, "asin": "A", "text_input": "alpha shampoo repair", "brand": "BrandOne", "categories": "Beauty Hair Care"},
                    {"item_id": 1, "asin": "B", "text_input": "beta serum vitamin c", "brand": "BrandTwo", "categories": "Beauty Skin Care"},
                    {"item_id": 2, "asin": "C", "text_input": "gamma shampoo nourish", "brand": "BrandOne", "categories": "Beauty Hair Care"},
                    {"item_id": 3, "asin": "D", "text_input": "delta lipstick matte", "brand": "BrandThree", "categories": "Beauty Makeup Lips"},
                ]
            ).to_pickle(items_path)

            torch.save(
                {
                    "item_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                    "text_embeddings": torch.randn(4, 2),
                    "image_embeddings": torch.randn(4, 3),
                    "text_available": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    "image_available": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    "asins": ["A", "B", "C", "D"],
                },
                source_path,
            )

            common_kwargs = dict(
                items_df_path=str(items_path),
                source_content_features_path=str(source_path),
                text_svd_dim=3,
                max_text_features=16,
                ngram_max=1,
                append_brand_embeddings=True,
                append_category_embeddings=True,
                brand_svd_dim=1,
                category_svd_dim=2,
            )
            build_metadata_content_features(output_path=str(base_output), **common_kwargs)
            build_metadata_content_features(
                output_path=str(scaled_output),
                brand_weight=3.0,
                category_weight=2.0,
                **common_kwargs,
            )

            base_payload = torch.load(base_output, map_location="cpu")
            scaled_payload = torch.load(scaled_output, map_location="cpu")
            self.assertTrue(torch.allclose(scaled_payload["text_embeddings"][:, -3:-2], base_payload["text_embeddings"][:, -3:-2] * 3.0, atol=1e-5))
            self.assertTrue(torch.allclose(scaled_payload["text_embeddings"][:, -2:], base_payload["text_embeddings"][:, -2:] * 2.0, atol=1e-5))

    def test_build_metadata_content_features_can_store_separate_structured_embeddings(self) -> None:
        try:
            import pandas as pd
            import torch
        except ImportError:
            self.skipTest("pandas/torch are not installed in the local Python environment")

        from mmsrec.features.metadata import build_metadata_content_features

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            items_path = root / "items.df"
            source_path = root / "source_content_features.pt"
            output_path = root / "metadata_structured_separate.pt"

            pd.DataFrame(
                [
                    {"item_id": 0, "asin": "A", "text_input": "alpha shampoo repair", "brand": "BrandOne", "categories": "Beauty Hair Care"},
                    {"item_id": 1, "asin": "B", "text_input": "beta serum vitamin c", "brand": "BrandTwo", "categories": "Beauty Skin Care"},
                    {"item_id": 2, "asin": "C", "text_input": "gamma shampoo nourish", "brand": "BrandOne", "categories": "Beauty Hair Care"},
                    {"item_id": 3, "asin": "D", "text_input": "delta lipstick matte", "brand": "", "categories": "Beauty Makeup Lips"},
                ]
            ).to_pickle(items_path)

            torch.save(
                {
                    "item_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
                    "text_embeddings": torch.randn(4, 2),
                    "image_embeddings": torch.randn(4, 3),
                    "text_available": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    "image_available": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    "asins": ["A", "B", "C", "D"],
                },
                source_path,
            )

            summary = build_metadata_content_features(
                items_df_path=str(items_path),
                source_content_features_path=str(source_path),
                output_path=str(output_path),
                text_svd_dim=3,
                max_text_features=16,
                ngram_max=1,
                append_brand_embeddings=True,
                append_category_embeddings=True,
                brand_svd_dim=1,
                category_svd_dim=2,
                concat_brand_into_text_embeddings=False,
                concat_category_into_text_embeddings=False,
                store_separate_structured_embeddings=True,
            )

            payload = torch.load(output_path, map_location="cpu")
            self.assertEqual(summary["text_embedding_dim"], 3)
            self.assertEqual(tuple(payload["text_embeddings"].shape), (4, 3))
            self.assertEqual(tuple(payload["brand_embeddings"].shape), (4, 1))
            self.assertEqual(tuple(payload["category_embeddings"].shape), (4, 2))
            self.assertTrue(torch.equal(payload["brand_available"], torch.tensor([1, 1, 1, 0], dtype=torch.int64)))
            self.assertTrue(torch.equal(payload["category_available"], torch.tensor([1, 1, 1, 1], dtype=torch.int64)))
            self.assertTrue(torch.allclose(payload["brand_embeddings"][0], payload["brand_embeddings"][2], atol=1e-5))
            self.assertTrue(torch.equal(payload["brand_embeddings"][3], torch.zeros(1, dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
