from __future__ import annotations

import json
import os
from pathlib import Path

from mmsrec.features.metadata import build_metadata_content_features
from mmsrec.multimodal.sasrec import MMTrainConfig, train_mm_sasrec


ROOT_DIR = Path("data/amazon_beauty_2013")
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ITEMS_PATH = ARTIFACTS_DIR / "items.df"
SOURCE_CONTENT_PATH_CANDIDATES = [
    ARTIFACTS_DIR / "content_features.pt",
]
if os.environ.get("MMSREC_SOURCE_CONTENT_PATH"):
    SOURCE_CONTENT_PATH_CANDIDATES.append(Path(os.environ["MMSREC_SOURCE_CONTENT_PATH"]))


def resolve_source_content_path() -> Path:
    for path in SOURCE_CONTENT_PATH_CANDIDATES:
        if path.exists():
            return path
    joined = ", ".join(str(path) for path in SOURCE_CONTENT_PATH_CANDIDATES)
    raise FileNotFoundError(f"could not find source content features in any candidate path: {joined}")


def build_feature_variants() -> list[dict]:
    source_content_path = resolve_source_content_path()
    variants = [
        {
            "name": "text_title2x",
            "feature_path": ARTIFACTS_DIR / "content_features_metadata_tfidf_df1_plus_source075_brand_category_w15_text_title2x.pt",
            "feature_kwargs": {
                "text_svd_dim": 512,
                "min_df": 1,
                "max_text_features": 50000,
                "ngram_max": 2,
                "append_source_text_embeddings": True,
                "append_brand_embeddings": True,
                "append_category_embeddings": True,
                "extra_title_repeats_in_text": 1,
                "brand_svd_dim": 32,
                "category_svd_dim": 64,
                "source_text_weight": 0.75,
                "brand_weight": 1.5,
                "category_weight": 1.5,
            },
        },
        {
            "name": "text_title3x",
            "feature_path": ARTIFACTS_DIR / "content_features_metadata_tfidf_df1_plus_source075_brand_category_w15_text_title3x.pt",
            "feature_kwargs": {
                "text_svd_dim": 512,
                "min_df": 1,
                "max_text_features": 50000,
                "ngram_max": 2,
                "append_source_text_embeddings": True,
                "append_brand_embeddings": True,
                "append_category_embeddings": True,
                "extra_title_repeats_in_text": 2,
                "brand_svd_dim": 32,
                "category_svd_dim": 64,
                "source_text_weight": 0.75,
                "brand_weight": 1.5,
                "category_weight": 1.5,
            },
        },
        {
            "name": "text_title2x_category2x",
            "feature_path": ARTIFACTS_DIR / "content_features_metadata_tfidf_df1_plus_source075_brand_category_w15_text_title2x_category2x.pt",
            "feature_kwargs": {
                "text_svd_dim": 512,
                "min_df": 1,
                "max_text_features": 50000,
                "ngram_max": 2,
                "append_source_text_embeddings": True,
                "append_brand_embeddings": True,
                "append_category_embeddings": True,
                "extra_title_repeats_in_text": 1,
                "extra_category_repeats_in_text": 1,
                "brand_svd_dim": 32,
                "category_svd_dim": 64,
                "source_text_weight": 0.75,
                "brand_weight": 1.5,
                "category_weight": 1.5,
            },
        },
        {
            "name": "text_title2x_brand2x_category2x",
            "feature_path": ARTIFACTS_DIR / "content_features_metadata_tfidf_df1_plus_source075_brand_category_w15_text_title2x_brand2x_category2x.pt",
            "feature_kwargs": {
                "text_svd_dim": 512,
                "min_df": 1,
                "max_text_features": 50000,
                "ngram_max": 2,
                "append_source_text_embeddings": True,
                "append_brand_embeddings": True,
                "append_category_embeddings": True,
                "extra_title_repeats_in_text": 1,
                "extra_category_repeats_in_text": 1,
                "extra_brand_repeats_in_text": 1,
                "brand_svd_dim": 32,
                "category_svd_dim": 64,
                "source_text_weight": 0.75,
                "brand_weight": 1.5,
                "category_weight": 1.5,
            },
        },
    ]

    for spec in variants:
        print(f"=== BUILD FEATURES {spec['name']} ===", flush=True)
        summary = build_metadata_content_features(
            items_df_path=str(ITEMS_PATH),
            source_content_features_path=str(source_content_path),
            output_path=str(spec["feature_path"]),
            **spec["feature_kwargs"],
        )
        spec["feature_summary"] = summary
        print(
            "=== BUILT FEATURES {name} text_dim={text_dim} output={output} ===".format(
                name=spec["name"],
                text_dim=summary["text_embedding_dim"],
                output=summary["output_path"],
            ),
            flush=True,
        )
    return variants


def run_training_variants(variants: list[dict]) -> list[dict]:
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
    for spec in variants:
        experiment_name = f"beauty2013_metadata_v26_{spec['name']}_full_tuned"
        print(f"=== START {experiment_name} ===", flush=True)
        config = MMTrainConfig(
            experiment_name=experiment_name,
            content_features_path=str(spec["feature_path"]),
            **base,
        )
        result = train_mm_sasrec(config)
        best_test = result["history"]["best_test_metrics"] or {}
        row = {
            "experiment_name": experiment_name,
            "feature_name": spec["name"],
            "feature_path": str(spec["feature_path"]),
            "feature_summary": spec["feature_summary"],
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
    variants = build_feature_variants()
    rows = run_training_variants(variants)
    output_path = Path("outputs/paper/beauty2013/beauty2013_weighted_text_v26_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
