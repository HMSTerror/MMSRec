from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmsrec.multimodal.sasrec import MMTrainConfig, train_mm_sasrec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one fixed Beauty2013 multimodal configuration on a chosen feature file.")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--content-features-path", required=True)
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = MMTrainConfig(
        root_dir="data/amazon_beauty_2013",
        content_features_path=args.content_features_path,
        checkpoint_dir="outputs/checkpoints",
        experiment_name=args.experiment_name,
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
    result = train_mm_sasrec(config)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
