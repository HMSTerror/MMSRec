import tempfile
import unittest
from pathlib import Path


class MultimodalSASRecTests(unittest.TestCase):
    def test_resolve_epoch_learning_rate_supports_cosine_warmup(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed in the local Python environment")

        from mmsrec.multimodal.sasrec import _resolve_epoch_learning_rate

        learning_rates = [
            _resolve_epoch_learning_rate(
                epoch=epoch,
                total_epochs=4,
                base_learning_rate=1e-3,
                scheduler_name="cosine",
                scheduler_min_lr=1e-4,
                scheduler_warmup_epochs=2,
                scheduler_warmup_start_factor=0.2,
            )
            for epoch in range(1, 5)
        ]

        self.assertAlmostEqual(learning_rates[0], 6e-4, places=10)
        self.assertAlmostEqual(learning_rates[1], 1e-3, places=10)
        self.assertAlmostEqual(learning_rates[2], 1e-3, places=10)
        self.assertAlmostEqual(learning_rates[3], 1e-4, places=10)

    def test_step_plateau_learning_rate_reduces_after_patience_is_exhausted(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed in the local Python environment")

        from mmsrec.multimodal.sasrec import _step_plateau_learning_rate

        current_lr = 1e-3
        best_metric = float("-inf")
        bad_epochs = 0

        current_lr, best_metric, bad_epochs = _step_plateau_learning_rate(
            current_learning_rate=current_lr,
            current_metric=0.5,
            best_metric=best_metric,
            bad_epoch_count=bad_epochs,
            factor=0.5,
            patience=1,
            min_learning_rate=1e-4,
            threshold=0.0,
        )
        self.assertAlmostEqual(current_lr, 1e-3, places=10)
        self.assertAlmostEqual(best_metric, 0.5, places=10)
        self.assertEqual(bad_epochs, 0)

        current_lr, best_metric, bad_epochs = _step_plateau_learning_rate(
            current_learning_rate=current_lr,
            current_metric=0.49,
            best_metric=best_metric,
            bad_epoch_count=bad_epochs,
            factor=0.5,
            patience=1,
            min_learning_rate=1e-4,
            threshold=0.0,
        )
        self.assertAlmostEqual(current_lr, 1e-3, places=10)
        self.assertAlmostEqual(best_metric, 0.5, places=10)
        self.assertEqual(bad_epochs, 1)

        current_lr, best_metric, bad_epochs = _step_plateau_learning_rate(
            current_learning_rate=current_lr,
            current_metric=0.48,
            best_metric=best_metric,
            bad_epoch_count=bad_epochs,
            factor=0.5,
            patience=1,
            min_learning_rate=1e-4,
            threshold=0.0,
        )
        self.assertAlmostEqual(current_lr, 5e-4, places=10)
        self.assertAlmostEqual(best_metric, 0.5, places=10)
        self.assertEqual(bad_epochs, 0)

    def test_cross_attention_block_handles_rows_with_all_masked_memory(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local Python environment")

        from mmsrec.multimodal.sasrec import CrossAttentionBlock

        block = CrossAttentionBlock(hidden_size=8, num_heads=2, dropout=0.1)
        query = torch.randn(2, 4, 8)
        memory = torch.randn(2, 4, 8)
        memory_mask = torch.tensor(
            [[False, False, False, False], [True, True, False, False]],
            dtype=torch.bool,
        )

        output = block(query, memory, memory_mask)

        self.assertTrue(torch.isfinite(output).all().item())
        self.assertTrue(torch.allclose(output[0], query[0], atol=1e-6))

    def test_load_content_features_appends_padding_row_and_masks(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local Python environment")

        from mmsrec.multimodal.sasrec import load_content_features

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "content_features.pt"
            torch.save(
                {
                    "item_ids": torch.tensor([0, 1, 2], dtype=torch.long),
                    "text_embeddings": torch.randn(3, 4),
                    "image_embeddings": torch.randn(3, 6),
                    "text_available": torch.tensor([1, 1, 0], dtype=torch.int64),
                    "image_available": torch.tensor([1, 0, 1], dtype=torch.int64),
                    "brand_embeddings": torch.randn(3, 2),
                    "category_embeddings": torch.randn(3, 3),
                    "brand_available": torch.tensor([1, 0, 1], dtype=torch.int64),
                    "category_available": torch.tensor([1, 1, 1], dtype=torch.int64),
                    "asins": ["A", "B", "C"],
                },
                path,
            )

            payload = load_content_features(str(path))

            self.assertEqual(tuple(payload["text_embeddings"].shape), (4, 4))
            self.assertEqual(tuple(payload["image_embeddings"].shape), (4, 6))
            self.assertEqual(tuple(payload["brand_embeddings"].shape), (4, 2))
            self.assertEqual(tuple(payload["category_embeddings"].shape), (4, 3))
            self.assertEqual(tuple(payload["text_available"].shape), (4,))
            self.assertEqual(tuple(payload["image_available"].shape), (4,))
            self.assertEqual(tuple(payload["brand_available"].shape), (4,))
            self.assertEqual(tuple(payload["category_available"].shape), (4,))
            self.assertEqual(int(payload["text_available"][-1].item()), 0)
            self.assertEqual(int(payload["image_available"][-1].item()), 0)
            self.assertEqual(int(payload["brand_available"][-1].item()), 0)
            self.assertEqual(int(payload["category_available"][-1].item()), 0)

    def test_mm_sasrec_forward_returns_item_logits(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local Python environment")

        from mmsrec.multimodal.sasrec import MMSASRec

        model = MMSASRec(
            item_num=5,
            seq_size=4,
            hidden_size=8,
            num_heads=2,
            num_blocks=1,
            dropout=0.1,
            padding_item_id=5,
            text_dim=4,
            image_dim=6,
            target_block_dim=4,
        )
        seq = torch.tensor([[0, 1, 5, 5], [2, 3, 4, 5]], dtype=torch.long)
        lengths = torch.tensor([2, 3], dtype=torch.long)
        text_embeddings = torch.randn(6, 4)
        image_embeddings = torch.randn(6, 6)
        text_available = torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.int64)
        image_available = torch.tensor([1, 0, 1, 1, 1, 0], dtype=torch.int64)

        logits = model(
            seq=seq,
            lengths=lengths,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            text_available=text_available,
            image_available=image_available,
        )

        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_mm_sasrec_forward_supports_modality_ablation(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local Python environment")

        from mmsrec.multimodal.sasrec import MMSASRec

        model = MMSASRec(
            item_num=5,
            seq_size=4,
            hidden_size=8,
            num_heads=2,
            num_blocks=1,
            dropout=0.1,
            padding_item_id=5,
            text_dim=4,
            image_dim=6,
            target_block_dim=4,
            use_text_modality=True,
            use_image_modality=False,
        )
        seq = torch.tensor([[0, 1, 5, 5], [2, 3, 4, 5]], dtype=torch.long)
        lengths = torch.tensor([2, 3], dtype=torch.long)
        text_embeddings = torch.randn(6, 4)
        image_embeddings = torch.randn(6, 6)
        text_available = torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.int64)
        image_available = torch.tensor([1, 0, 1, 1, 1, 0], dtype=torch.int64)

        logits = model(
            seq=seq,
            lengths=lengths,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            text_available=text_available,
            image_available=image_available,
        )

        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_mm_sasrec_forward_supports_separate_sequence_fusion(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local Python environment")

        from mmsrec.multimodal.sasrec import MMSASRec

        model = MMSASRec(
            item_num=5,
            seq_size=4,
            hidden_size=8,
            num_heads=2,
            num_blocks=1,
            dropout=0.1,
            padding_item_id=5,
            text_dim=4,
            image_dim=6,
            target_block_dim=4,
            use_text_modality=True,
            use_image_modality=True,
            separate_sequence_fusion=True,
        )
        seq = torch.tensor([[0, 1, 5, 5], [2, 3, 4, 5]], dtype=torch.long)
        lengths = torch.tensor([2, 3], dtype=torch.long)
        text_embeddings = torch.randn(6, 4)
        image_embeddings = torch.randn(6, 6)
        text_available = torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.int64)
        image_available = torch.tensor([1, 0, 1, 1, 1, 0], dtype=torch.int64)

        logits = model(
            seq=seq,
            lengths=lengths,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            text_available=text_available,
            image_available=image_available,
        )

        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_mm_sasrec_forward_supports_explicit_structured_branches(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local Python environment")

        from mmsrec.multimodal.sasrec import MMSASRec

        model = MMSASRec(
            item_num=5,
            seq_size=4,
            hidden_size=8,
            num_heads=2,
            num_blocks=1,
            dropout=0.1,
            padding_item_id=5,
            text_dim=4,
            image_dim=6,
            brand_dim=2,
            category_dim=3,
            target_block_dim=4,
            use_text_modality=True,
            use_image_modality=True,
            use_brand_branch=True,
            use_category_branch=True,
        )
        seq = torch.tensor([[0, 1, 5, 5], [2, 3, 4, 5]], dtype=torch.long)
        lengths = torch.tensor([2, 3], dtype=torch.long)
        text_embeddings = torch.randn(6, 4)
        image_embeddings = torch.randn(6, 6)
        brand_embeddings = torch.randn(6, 2)
        category_embeddings = torch.randn(6, 3)
        text_available = torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.int64)
        image_available = torch.tensor([1, 0, 1, 1, 1, 0], dtype=torch.int64)
        brand_available = torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.int64)
        category_available = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.int64)

        logits = model(
            seq=seq,
            lengths=lengths,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            brand_embeddings=brand_embeddings,
            category_embeddings=category_embeddings,
            text_available=text_available,
            image_available=image_available,
            brand_available=brand_available,
            category_available=category_available,
        )

        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_mm_sasrec_forward_supports_category_profile_without_category_score_branch(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local Python environment")

        from mmsrec.multimodal.sasrec import MMSASRec

        model = MMSASRec(
            item_num=5,
            seq_size=4,
            hidden_size=8,
            num_heads=2,
            num_blocks=1,
            dropout=0.1,
            padding_item_id=5,
            text_dim=4,
            image_dim=6,
            category_dim=3,
            target_block_dim=4,
            use_text_modality=True,
            use_image_modality=True,
            use_category_branch=True,
            use_category_fusion_branch=False,
            use_category_score_branch=False,
            use_category_profile=True,
        )
        seq = torch.tensor([[0, 1, 5, 5], [2, 3, 4, 5]], dtype=torch.long)
        lengths = torch.tensor([2, 3], dtype=torch.long)
        text_embeddings = torch.randn(6, 4)
        image_embeddings = torch.randn(6, 6)
        category_embeddings = torch.randn(6, 3)
        text_available = torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.int64)
        image_available = torch.tensor([1, 0, 1, 1, 1, 0], dtype=torch.int64)
        category_available = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.int64)

        logits = model(
            seq=seq,
            lengths=lengths,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            category_embeddings=category_embeddings,
            text_available=text_available,
            image_available=image_available,
            category_available=category_available,
        )

        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_mm_sasrec_forward_supports_category_memory_without_candidate_category_branch(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local Python environment")

        from mmsrec.multimodal.sasrec import MMSASRec

        model = MMSASRec(
            item_num=5,
            seq_size=4,
            hidden_size=8,
            num_heads=2,
            num_blocks=1,
            dropout=0.1,
            padding_item_id=5,
            text_dim=4,
            image_dim=6,
            category_dim=3,
            target_block_dim=4,
            use_text_modality=True,
            use_image_modality=True,
            use_category_branch=False,
            use_category_memory=True,
        )
        seq = torch.tensor([[0, 1, 5, 5], [2, 3, 4, 5]], dtype=torch.long)
        lengths = torch.tensor([2, 3], dtype=torch.long)
        text_embeddings = torch.randn(6, 4)
        image_embeddings = torch.randn(6, 6)
        category_embeddings = torch.randn(6, 3)
        text_available = torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.int64)
        image_available = torch.tensor([1, 0, 1, 1, 1, 0], dtype=torch.int64)
        category_available = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.int64)

        logits = model(
            seq=seq,
            lengths=lengths,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            category_embeddings=category_embeddings,
            text_available=text_available,
            image_available=image_available,
            category_available=category_available,
        )

        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertTrue(torch.isfinite(logits).all().item())

    def test_mm_sasrec_rejects_structured_memory_with_separate_sequence_fusion(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch is not installed in the local Python environment")

        from mmsrec.multimodal.sasrec import MMSASRec

        with self.assertRaisesRegex(ValueError, "structured memory branches are not supported"):
            MMSASRec(
                item_num=5,
                seq_size=4,
                hidden_size=8,
                num_heads=2,
                num_blocks=1,
                dropout=0.1,
                padding_item_id=5,
                text_dim=4,
                image_dim=6,
                category_dim=3,
                target_block_dim=4,
                use_category_memory=True,
                separate_sequence_fusion=True,
            )

    def test_mm_sasrec_separate_sequence_fusion_handles_rows_with_no_valid_image_memory(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in the local Python environment")

        from mmsrec.multimodal.sasrec import MMSASRec

        model = MMSASRec(
            item_num=5,
            seq_size=4,
            hidden_size=8,
            num_heads=2,
            num_blocks=1,
            dropout=0.1,
            padding_item_id=5,
            text_dim=4,
            image_dim=6,
            target_block_dim=4,
            use_text_modality=True,
            use_image_modality=True,
            separate_sequence_fusion=True,
        )
        seq = torch.tensor([[3, 5, 5, 5], [2, 3, 4, 5]], dtype=torch.long)
        lengths = torch.tensor([1, 3], dtype=torch.long)
        text_embeddings = torch.randn(6, 4)
        image_embeddings = torch.randn(6, 6)
        text_available = torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.int64)
        image_available = torch.tensor([1, 1, 1, 0, 1, 0], dtype=torch.int64)

        logits = model(
            seq=seq,
            lengths=lengths,
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            text_available=text_available,
            image_available=image_available,
        )

        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
