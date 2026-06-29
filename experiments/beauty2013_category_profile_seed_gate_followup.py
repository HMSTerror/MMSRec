from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmsrec.multimodal.sasrec import MMTrainConfig, train_mm_sasrec


ROOT_DIR = Path("data/amazon_beauty_2013")
FEATURE_PATH = ROOT_DIR / "artifacts" / "content_features_metadata_source075_w15_with_category_profile_only.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run seed-gate follow-up for the strongest category-profile family."
    )
    parser.add_argument("--feature-path", default=str(FEATURE_PATH))
    parser.add_argument("--experiment-tag", default="v48_profile_seed_gate")
    parser.add_argument("--seeds", default="101,103")
    parser.add_argument("--gates", default="-3.0,-2.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    gates = [float(part.strip()) for part in args.gates.split(",") if part.strip()]
    rows = []
    for seed in seeds:
        for gate in gates:
            gate_slug = str(gate).replace("-", "m").replace(".", "p")
            experiment_name = f"beauty2013_metadata_{args.experiment_tag}_seed{seed}_{gate_slug}"
            print(f"=== START {experiment_name} ===", flush=True)
            config = MMTrainConfig(
                root_dir=str(ROOT_DIR),
                content_features_path=args.feature_path,
                checkpoint_dir="outputs/checkpoints",
                experiment_name=experiment_name,
                device="cuda",
                epochs=14,
                batch_size=256,
                eval_batch_size=512,
                learning_rate=5e-4,
                weight_decay=1e-4,
                random_seed=seed,
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
                use_brand_fusion_branch=False,
                use_category_fusion_branch=False,
                use_brand_score_branch=False,
                use_category_score_branch=False,
                use_brand_profile=False,
                use_category_profile=True,
                use_brand_memory=False,
                use_category_memory=False,
                separate_sequence_fusion=False,
                id_score_gate_init=-1.2,
                text_score_gate_init=-0.35,
                image_score_gate_init=-3.2,
                brand_score_gate_init=-2.0,
                category_score_gate_init=-2.0,
                text_profile_gate_init=-8.0,
                image_profile_gate_init=-8.0,
                brand_profile_gate_init=-3.0,
                category_profile_gate_init=gate,
            )
            result = train_mm_sasrec(config)
            best_test = result["history"]["best_test_metrics"] or {}
            row = {
                "experiment_name": experiment_name,
                "feature_path": args.feature_path,
                "seed": seed,
                "category_profile_gate_init": gate,
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

    output_path = Path("outputs/paper/beauty2013") / f"{args.experiment_tag}_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
