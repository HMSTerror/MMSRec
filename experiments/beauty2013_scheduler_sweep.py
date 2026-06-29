from __future__ import annotations

import json
from pathlib import Path

from mmsrec.multimodal.sasrec import MMTrainConfig, train_mm_sasrec


def main() -> None:
    base = dict(
        root_dir="data/amazon_beauty_2013",
        content_features_path="data/amazon_beauty_2013/artifacts/content_features_metadata_tfidf_df1_plus_source075_brand_category_w15.pt",
        checkpoint_dir="outputs/checkpoints",
        device="cuda",
        batch_size=256,
        eval_batch_size=512,
        learning_rate=5e-4,
        weight_decay=1e-4,
        random_seed=100,
        hidden_size=192,
        num_heads=6,
        num_blocks=2,
        dropout=0.15,
        topk=(10, 20, 50),
        early_stop_metric="NDCG@10",
        early_stop_min_delta=0.0,
        target_block_dim=96,
        use_text_modality=True,
        use_image_modality=True,
        use_brand_branch=False,
        use_category_branch=False,
        separate_sequence_fusion=False,
        id_score_gate_init=-1.2,
        text_score_gate_init=-0.35,
        image_score_gate_init=-3.2,
        brand_score_gate_init=-2.0,
        category_score_gate_init=-2.0,
        text_profile_gate_init=-8.0,
        image_profile_gate_init=-8.0,
    )
    variants = [
        (
            "beauty2013_metadata_v23_source075_w15_cosine_w1_min5e5_ep20",
            dict(
                epochs=20,
                early_stop_patience=8,
                lr_scheduler="cosine",
                lr_scheduler_min_lr=5e-5,
                lr_scheduler_warmup_epochs=1,
                lr_scheduler_warmup_start_factor=0.2,
            ),
        ),
        (
            "beauty2013_metadata_v23_source075_w15_cosine_w2_min5e5_ep20",
            dict(
                epochs=20,
                early_stop_patience=8,
                lr_scheduler="cosine",
                lr_scheduler_min_lr=5e-5,
                lr_scheduler_warmup_epochs=2,
                lr_scheduler_warmup_start_factor=0.2,
            ),
        ),
        (
            "beauty2013_metadata_v23_source075_w15_cosine_w1_min1e4_ep20",
            dict(
                epochs=20,
                early_stop_patience=8,
                lr_scheduler="cosine",
                lr_scheduler_min_lr=1e-4,
                lr_scheduler_warmup_epochs=1,
                lr_scheduler_warmup_start_factor=0.2,
            ),
        ),
        (
            "beauty2013_metadata_v23_source075_w15_cosine_w0_min0_ep18",
            dict(
                epochs=18,
                early_stop_patience=7,
                lr_scheduler="cosine",
                lr_scheduler_min_lr=0.0,
                lr_scheduler_warmup_epochs=0,
                lr_scheduler_warmup_start_factor=0.2,
            ),
        ),
    ]

    rows = []
    for experiment_name, overrides in variants:
        print(f"=== START {experiment_name} ===", flush=True)
        config = MMTrainConfig(
            experiment_name=experiment_name,
            **base,
            **overrides,
        )
        result = train_mm_sasrec(config)
        best_test = result["history"]["best_test_metrics"] or {}
        row = {
            "experiment_name": experiment_name,
            "best_epoch": result["history"]["best_epoch"],
            "stop_epoch": result["history"]["stop_epoch"],
            "best_test_ndcg10": best_test.get("NDCG@10"),
            "best_test_hr10": best_test.get("HR@10"),
            "learning_rates": result["history"]["learning_rates"],
            "config": result["config"],
        }
        rows.append(row)
        print(
            "=== DONE {name} epoch={epoch} ndcg10={ndcg} hr10={hr10} stop_epoch={stop_epoch} ===".format(
                name=experiment_name,
                epoch=row["best_epoch"],
                ndcg=row["best_test_ndcg10"],
                hr10=row["best_test_hr10"],
                stop_epoch=row["stop_epoch"],
            ),
            flush=True,
        )

    output_path = Path("outputs/paper/beauty2013/beauty2013_scheduler_v23_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
