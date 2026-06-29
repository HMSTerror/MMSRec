from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmsrec.baselines.sasrec import BaselineTrainConfig, train_sasrec_baseline


ROOT_DIR = Path("data/amazon_beauty_2013")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-seed follow-up for the PyTorch SASRec baseline."
    )
    parser.add_argument("--root-dir", default=str(ROOT_DIR))
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiment-tag", default="v46_baseline_seed")
    parser.add_argument("--seeds", default="101,102")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    rows = []
    for seed in seeds:
        experiment_name = f"beauty2013_paper_{args.experiment_tag}_seed{seed}"
        print(f"=== START {experiment_name} ===", flush=True)
        config = BaselineTrainConfig(
            root_dir=args.root_dir,
            checkpoint_dir=args.checkpoint_dir,
            experiment_name=experiment_name,
            device=args.device,
            epochs=10,
            batch_size=512,
            eval_batch_size=512,
            learning_rate=1e-3,
            weight_decay=0.0,
            random_seed=seed,
            hidden_size=64,
            num_heads=1,
            num_blocks=1,
            dropout=0.1,
            topk=(10, 20, 50),
            early_stop_metric="NDCG@10",
            early_stop_patience=3,
            early_stop_min_delta=0.0,
        )
        result = train_sasrec_baseline(config)
        best_test = result["history"]["best_test_metrics"] or {}
        row = {
            "experiment_name": experiment_name,
            "seed": seed,
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
