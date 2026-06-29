from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch

from mmsrec.features.metadata import _build_text_embeddings, _safe_text
from mmsrec.multimodal.sasrec import MMTrainConfig, train_mm_sasrec


ROOT_DIR = Path("data/amazon_beauty_2013")
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ITEMS_PATH = ARTIFACTS_DIR / "items.df"
BASE_FEATURE_PATH = ARTIFACTS_DIR / "content_features_metadata_tfidf_df1_plus_source075_brand_category_w15.pt"


def build_feature_variants() -> list[dict]:
    items = pd.read_pickle(ITEMS_PATH).sort_values("item_id").reset_index(drop=True)
    payload = torch.load(BASE_FEATURE_PATH, map_location="cpu")

    category_texts = [_safe_text(value) for value in items.get("categories", pd.Series([""] * len(items))).tolist()]
    category_available = torch.tensor([1 if text else 0 for text in category_texts], dtype=torch.int64)
    raw_category_embeddings = _build_text_embeddings(
        category_texts,
        text_svd_dim=64,
        min_df=1,
        max_text_features=50000,
        ngram_max=2,
        random_state=100,
        stop_words=None,
    )
    raw_category_embeddings = raw_category_embeddings * category_available.unsqueeze(1).numpy().astype(raw_category_embeddings.dtype)

    variants = []
    for weight in (1.0, 1.5, 2.0):
        weight_slug = str(weight).replace(".", "p")
        variant_payload = deepcopy(payload)
        category_embeddings = raw_category_embeddings * float(weight)
        variant_payload["category_embeddings"] = torch.from_numpy(category_embeddings).to(dtype=torch.float32)
        variant_payload["category_available"] = category_available
        output_path = ARTIFACTS_DIR / f"content_features_metadata_source075_w15_with_category_memory_w{weight_slug}.pt"
        torch.save(variant_payload, output_path)
        variants.append(
            {
                "name": f"catmemory_w{weight_slug}",
                "weight": weight,
                "feature_path": output_path,
            }
        )
        print(f"=== BUILT FEATURES {output_path} weight={weight} ===", flush=True)
    return variants


def main() -> None:
    variants = build_feature_variants()
    rows = []
    for feature in variants:
        experiment_name = f"beauty2013_metadata_v37_source075_w15_{feature['name']}"
        print(f"=== START {experiment_name} ===", flush=True)
        config = MMTrainConfig(
            root_dir=str(ROOT_DIR),
            content_features_path=str(feature["feature_path"]),
            checkpoint_dir="outputs/checkpoints",
            experiment_name=experiment_name,
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
            use_brand_fusion_branch=False,
            use_category_fusion_branch=False,
            use_brand_score_branch=False,
            use_category_score_branch=False,
            use_brand_profile=False,
            use_category_profile=False,
            use_brand_memory=False,
            use_category_memory=True,
            separate_sequence_fusion=False,
            id_score_gate_init=-1.2,
            text_score_gate_init=-0.35,
            image_score_gate_init=-3.2,
            brand_score_gate_init=-2.0,
            category_score_gate_init=-2.0,
            text_profile_gate_init=-8.0,
            image_profile_gate_init=-8.0,
            brand_profile_gate_init=-3.0,
            category_profile_gate_init=-2.0,
        )
        result = train_mm_sasrec(config)
        best_test = result["history"]["best_test_metrics"] or {}
        row = {
            "experiment_name": experiment_name,
            "feature_path": str(feature["feature_path"]),
            "category_memory_weight": feature["weight"],
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

    output_path = Path("outputs/paper/beauty2013/beauty2013_category_memory_v37_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
