from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import torch

from mmsrec.multimodal.sasrec import MMTrainConfig, train_mm_sasrec


ROOT_DIR = Path("data/amazon_beauty_2013")
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
BASE_FEATURE_PATH = ARTIFACTS_DIR / "content_features_metadata_tfidf_df1_plus_source075_brand_category_w15_text_category2x.pt"
SOURCE_TEXT_DIM = 768
BRAND_DIM = 32
CATEGORY_DIM = 64
BASE_SOURCE_WEIGHT = 0.75
BASE_BRAND_WEIGHT = 1.5
BASE_CATEGORY_WEIGHT = 1.5


def build_rescaled_feature_variants() -> list[dict]:
    payload = torch.load(BASE_FEATURE_PATH, map_location="cpu")
    text_embeddings = payload["text_embeddings"].clone()
    total_dim = int(text_embeddings.shape[1])
    base_text_dim = total_dim - SOURCE_TEXT_DIM - BRAND_DIM - CATEGORY_DIM
    if base_text_dim <= 0:
        raise ValueError(f"invalid base_text_dim derived from total_dim={total_dim}")

    source_slice = slice(base_text_dim, base_text_dim + SOURCE_TEXT_DIM)
    brand_slice = slice(base_text_dim + SOURCE_TEXT_DIM, base_text_dim + SOURCE_TEXT_DIM + BRAND_DIM)
    category_slice = slice(base_text_dim + SOURCE_TEXT_DIM + BRAND_DIM, total_dim)

    variants = [
        {
            "name": "text_category2x_cw125_fast",
            "output_path": ARTIFACTS_DIR / "content_features_metadata_fast_text_category2x_cw125.pt",
            "source_weight": 0.75,
            "brand_weight": 1.5,
            "category_weight": 1.25,
        },
        {
            "name": "text_category2x_cw100_fast",
            "output_path": ARTIFACTS_DIR / "content_features_metadata_fast_text_category2x_cw100.pt",
            "source_weight": 0.75,
            "brand_weight": 1.5,
            "category_weight": 1.0,
        },
        {
            "name": "text_category2x_source072_cw125_fast",
            "output_path": ARTIFACTS_DIR / "content_features_metadata_fast_text_category2x_source072_cw125.pt",
            "source_weight": 0.72,
            "brand_weight": 1.5,
            "category_weight": 1.25,
        },
        {
            "name": "text_category2x_source078_cw125_fast",
            "output_path": ARTIFACTS_DIR / "content_features_metadata_fast_text_category2x_source078_cw125.pt",
            "source_weight": 0.78,
            "brand_weight": 1.5,
            "category_weight": 1.25,
        },
        {
            "name": "text_category2x_source080_cw125_fast",
            "output_path": ARTIFACTS_DIR / "content_features_metadata_fast_text_category2x_source080_cw125.pt",
            "source_weight": 0.80,
            "brand_weight": 1.5,
            "category_weight": 1.25,
        },
    ]

    rows = []
    for spec in variants:
        variant_payload = deepcopy(payload)
        variant_embeddings = text_embeddings.clone()
        variant_embeddings[:, source_slice] *= float(spec["source_weight"] / BASE_SOURCE_WEIGHT)
        variant_embeddings[:, brand_slice] *= float(spec["brand_weight"] / BASE_BRAND_WEIGHT)
        variant_embeddings[:, category_slice] *= float(spec["category_weight"] / BASE_CATEGORY_WEIGHT)
        variant_payload["text_embeddings"] = variant_embeddings
        torch.save(variant_payload, spec["output_path"])
        row = {
            "name": spec["name"],
            "output_path": str(spec["output_path"]),
            "text_embedding_dim": total_dim,
            "source_weight": spec["source_weight"],
            "brand_weight": spec["brand_weight"],
            "category_weight": spec["category_weight"],
        }
        rows.append(row)
        print(
            "=== BUILT FEATURES {name} dim={dim} source={source} brand={brand} category={category} ===".format(
                name=row["name"],
                dim=row["text_embedding_dim"],
                source=row["source_weight"],
                brand=row["brand_weight"],
                category=row["category_weight"],
            ),
            flush=True,
        )
    return rows


def run_training_variants(feature_rows: list[dict]) -> list[dict]:
    base = dict(
        root_dir=str(ROOT_DIR),
        checkpoint_dir="outputs/checkpoints",
        device="cuda",
        epochs=14,
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
        early_stop_patience=5,
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

    rows = []
    for feature in feature_rows:
        experiment_name = f"beauty2013_metadata_v31_{feature['name']}_full_tuned"
        print(f"=== START {experiment_name} ===", flush=True)
        config = MMTrainConfig(
            experiment_name=experiment_name,
            content_features_path=feature["output_path"],
            **base,
        )
        result = train_mm_sasrec(config)
        best_test = result["history"]["best_test_metrics"] or {}
        row = {
            "experiment_name": experiment_name,
            "feature_name": feature["name"],
            "feature_path": feature["output_path"],
            "feature_summary": feature,
            "best_epoch": result["history"]["best_epoch"],
            "stop_epoch": result["history"]["stop_epoch"],
            "best_test_ndcg10": best_test.get("NDCG@10"),
            "best_test_hr10": best_test.get("HR@10"),
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
    return rows


def main() -> None:
    feature_rows = build_rescaled_feature_variants()
    rows = run_training_variants(feature_rows)
    output_path = Path("outputs/paper/beauty2013/beauty2013_category_transform_v31_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
