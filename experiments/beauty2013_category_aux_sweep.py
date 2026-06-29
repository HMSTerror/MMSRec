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


def build_feature_file() -> Path:
    items = pd.read_pickle(ITEMS_PATH).sort_values("item_id").reset_index(drop=True)
    payload = torch.load(BASE_FEATURE_PATH, map_location="cpu")

    category_texts = [_safe_text(value) for value in items.get("categories", pd.Series([""] * len(items))).tolist()]
    category_available = torch.tensor([1 if text else 0 for text in category_texts], dtype=torch.int64)
    category_embeddings = _build_text_embeddings(
        category_texts,
        text_svd_dim=64,
        min_df=1,
        max_text_features=50000,
        ngram_max=2,
        random_state=100,
        stop_words=None,
    )
    category_embeddings = category_embeddings * 1.5
    category_embeddings = category_embeddings * category_available.unsqueeze(1).numpy().astype(category_embeddings.dtype)

    variant_payload = deepcopy(payload)
    variant_payload["category_embeddings"] = torch.from_numpy(category_embeddings).to(dtype=torch.float32)
    variant_payload["category_available"] = category_available

    output_path = ARTIFACTS_DIR / "content_features_metadata_source075_w15_with_category_aux.pt"
    torch.save(variant_payload, output_path)
    print(f"=== BUILT FEATURES {output_path} ===", flush=True)
    return output_path


def build_base_config(*, feature_path: Path, experiment_name: str) -> MMTrainConfig:
    return MMTrainConfig(
        root_dir=str(ROOT_DIR),
        content_features_path=str(feature_path),
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
        use_category_branch=True,
        use_brand_score_branch=False,
        use_category_score_branch=False,
        use_brand_profile=False,
        use_category_profile=False,
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


def main() -> None:
    feature_path = build_feature_file()
    variants = [
        {
            "name": "catfusion_only",
            "use_category_profile": False,
            "category_profile_gate_init": -2.0,
        },
        {
            "name": "catfusion_catprofile_m2p5",
            "use_category_profile": True,
            "category_profile_gate_init": -2.5,
        },
        {
            "name": "catfusion_catprofile_m2p0",
            "use_category_profile": True,
            "category_profile_gate_init": -2.0,
        },
        {
            "name": "catfusion_catprofile_m1p5",
            "use_category_profile": True,
            "category_profile_gate_init": -1.5,
        },
    ]

    rows = []
    for spec in variants:
        experiment_name = f"beauty2013_metadata_v35_source075_w15_{spec['name']}"
        print(f"=== START {experiment_name} ===", flush=True)
        config = build_base_config(feature_path=feature_path, experiment_name=experiment_name)
        config.use_category_profile = bool(spec["use_category_profile"])
        config.category_profile_gate_init = float(spec["category_profile_gate_init"])

        result = train_mm_sasrec(config)
        best_test = result["history"]["best_test_metrics"] or {}
        row = {
            "experiment_name": experiment_name,
            "feature_path": str(feature_path),
            "variant": spec["name"],
            "use_category_profile": config.use_category_profile,
            "category_profile_gate_init": config.category_profile_gate_init,
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

    output_path = Path("outputs/paper/beauty2013/beauty2013_category_aux_v35_summary.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
