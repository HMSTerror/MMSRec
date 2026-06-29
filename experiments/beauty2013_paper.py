from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from mmsrec.baselines.sasrec import BaselineTrainConfig, train_sasrec_baseline
from mmsrec.multimodal.sasrec import MMTrainConfig, train_mm_sasrec


ROOT_DIR = "data/amazon_beauty_2013"
CHECKPOINT_DIR = "outputs/checkpoints"
DEFAULT_CONTENT_FEATURES_PATH = "data/amazon_beauty_2013/artifacts/content_features.pt"


def build_baseline_config(*, root_dir: str, checkpoint_dir: str, device: str, experiment_name: str) -> BaselineTrainConfig:
    return BaselineTrainConfig(
        root_dir=root_dir,
        checkpoint_dir=checkpoint_dir,
        experiment_name=experiment_name,
        device=device,
        epochs=10,
        batch_size=512,
        eval_batch_size=512,
        learning_rate=1e-3,
        weight_decay=0.0,
        random_seed=100,
        hidden_size=64,
        num_heads=1,
        num_blocks=1,
        dropout=0.1,
        topk=(10, 20, 50),
        early_stop_metric="NDCG@10",
        early_stop_patience=3,
        early_stop_min_delta=0.0,
    )


def build_mm_default_config(
    *,
    root_dir: str,
    checkpoint_dir: str,
    content_features_path: str,
    device: str,
    experiment_name: str,
) -> MMTrainConfig:
    return MMTrainConfig(
        root_dir=root_dir,
        content_features_path=content_features_path,
        checkpoint_dir=checkpoint_dir,
        experiment_name=experiment_name,
        device=device,
        epochs=12,
        batch_size=512,
        eval_batch_size=512,
        learning_rate=1e-3,
        weight_decay=0.0,
        random_seed=100,
        hidden_size=128,
        num_heads=4,
        num_blocks=2,
        dropout=0.1,
        topk=(10, 20, 50),
        early_stop_metric="NDCG@10",
        early_stop_patience=4,
        early_stop_min_delta=0.0,
        target_block_dim=64,
        use_text_modality=True,
        use_image_modality=True,
    )


def build_mm_tuned_config(
    *,
    root_dir: str,
    checkpoint_dir: str,
    content_features_path: str,
    device: str,
    experiment_name: str,
) -> MMTrainConfig:
    return MMTrainConfig(
        root_dir=root_dir,
        content_features_path=content_features_path,
        checkpoint_dir=checkpoint_dir,
        experiment_name=experiment_name,
        device=device,
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
    )


def build_mm_text_only_config(
    *,
    root_dir: str,
    checkpoint_dir: str,
    content_features_path: str,
    device: str,
    experiment_name: str,
) -> MMTrainConfig:
    config = build_mm_default_config(
        root_dir=root_dir,
        checkpoint_dir=checkpoint_dir,
        content_features_path=content_features_path,
        device=device,
        experiment_name=experiment_name,
    )
    config.use_image_modality = False
    return config


def build_mm_text_biased_config(
    *,
    root_dir: str,
    checkpoint_dir: str,
    content_features_path: str,
    device: str,
    experiment_name: str,
) -> MMTrainConfig:
    config = build_mm_default_config(
        root_dir=root_dir,
        checkpoint_dir=checkpoint_dir,
        content_features_path=content_features_path,
        device=device,
        experiment_name=experiment_name,
    )
    config.id_score_gate_init = -1.5
    config.text_score_gate_init = -0.5
    config.image_score_gate_init = -3.0
    config.text_profile_gate_init = -0.4
    config.image_profile_gate_init = -3.2
    return config


def build_mm_text_biased_tuned_config(
    *,
    root_dir: str,
    checkpoint_dir: str,
    content_features_path: str,
    device: str,
    experiment_name: str,
) -> MMTrainConfig:
    config = build_mm_tuned_config(
        root_dir=root_dir,
        checkpoint_dir=checkpoint_dir,
        content_features_path=content_features_path,
        device=device,
        experiment_name=experiment_name,
    )
    config.id_score_gate_init = -1.2
    config.text_score_gate_init = -0.35
    config.image_score_gate_init = -3.2
    config.text_profile_gate_init = -0.25
    config.image_profile_gate_init = -3.5
    return config


def build_mm_image_only_config(
    *,
    root_dir: str,
    checkpoint_dir: str,
    content_features_path: str,
    device: str,
    experiment_name: str,
) -> MMTrainConfig:
    config = build_mm_default_config(
        root_dir=root_dir,
        checkpoint_dir=checkpoint_dir,
        content_features_path=content_features_path,
        device=device,
        experiment_name=experiment_name,
    )
    config.use_text_modality = False
    return config


def build_mm_seqsplit_default_config(
    *,
    root_dir: str,
    checkpoint_dir: str,
    content_features_path: str,
    device: str,
    experiment_name: str,
) -> MMTrainConfig:
    config = build_mm_default_config(
        root_dir=root_dir,
        checkpoint_dir=checkpoint_dir,
        content_features_path=content_features_path,
        device=device,
        experiment_name=experiment_name,
    )
    config.separate_sequence_fusion = True
    config.id_score_gate_init = -1.5
    config.text_score_gate_init = -0.5
    config.image_score_gate_init = -3.0
    config.text_profile_gate_init = -8.0
    config.image_profile_gate_init = -8.0
    return config


def build_mm_seqsplit_tuned_config(
    *,
    root_dir: str,
    checkpoint_dir: str,
    content_features_path: str,
    device: str,
    experiment_name: str,
) -> MMTrainConfig:
    config = build_mm_tuned_config(
        root_dir=root_dir,
        checkpoint_dir=checkpoint_dir,
        content_features_path=content_features_path,
        device=device,
        experiment_name=experiment_name,
    )
    config.separate_sequence_fusion = True
    config.id_score_gate_init = -1.2
    config.text_score_gate_init = -0.35
    config.image_score_gate_init = -3.2
    config.text_profile_gate_init = -8.0
    config.image_profile_gate_init = -8.0
    return config


VARIANTS: dict[str, dict[str, object]] = {
    "baseline": {
        "family": "baseline",
        "builder": build_baseline_config,
    },
    "mm_decomp_full_default": {
        "family": "multimodal",
        "builder": build_mm_default_config,
    },
    "mm_decomp_full_tuned": {
        "family": "multimodal",
        "builder": build_mm_tuned_config,
    },
    "mm_decomp_full_text_biased": {
        "family": "multimodal",
        "builder": build_mm_text_biased_config,
    },
    "mm_decomp_full_text_biased_tuned": {
        "family": "multimodal",
        "builder": build_mm_text_biased_tuned_config,
    },
    "mm_decomp_text_only": {
        "family": "multimodal",
        "builder": build_mm_text_only_config,
    },
    "mm_decomp_image_only": {
        "family": "multimodal",
        "builder": build_mm_image_only_config,
    },
    "mm_seqsplit_default": {
        "family": "multimodal",
        "builder": build_mm_seqsplit_default_config,
    },
    "mm_seqsplit_tuned": {
        "family": "multimodal",
        "builder": build_mm_seqsplit_tuned_config,
    },
}


def run_variant(
    name: str,
    *,
    root_dir: str,
    checkpoint_dir: str,
    content_features_path: str,
    device: str,
    experiment_prefix: str,
) -> dict:
    spec = VARIANTS[name]
    experiment_name = f"{experiment_prefix}_{name}"
    builder: Callable[..., BaselineTrainConfig | MMTrainConfig] = spec["builder"]  # type: ignore[assignment]
    if spec["family"] == "baseline":
        config = builder(
            root_dir=root_dir,
            checkpoint_dir=checkpoint_dir,
            device=device,
            experiment_name=experiment_name,
        )
        result = train_sasrec_baseline(config)
    else:
        config = builder(
            root_dir=root_dir,
            checkpoint_dir=checkpoint_dir,
            content_features_path=content_features_path,
            device=device,
            experiment_name=experiment_name,
        )
        result = train_mm_sasrec(config)

    return summarize_result(name=name, family=str(spec["family"]), result=result)


def summarize_result(*, name: str, family: str, result: dict) -> dict:
    history = result["history"]
    config = result["config"]
    best_val = history["best_val_metrics"]
    best_test = history["best_test_metrics"]
    return {
        "variant": name,
        "family": family,
        "experiment_name": config["experiment_name"],
        "checkpoint_path": result["checkpoint_path"],
        "best_epoch": history["best_epoch"],
        "stop_epoch": history["stop_epoch"],
        "best_val_ndcg10": None if best_val is None else best_val["NDCG@10"],
        "best_test_ndcg10": None if best_test is None else best_test["NDCG@10"],
        "best_test_hr10": None if best_test is None else best_test["HR@10"],
        "use_text_modality": config.get("use_text_modality", False),
        "use_image_modality": config.get("use_image_modality", False),
        "use_brand_branch": config.get("use_brand_branch", False),
        "use_category_branch": config.get("use_category_branch", False),
        "use_brand_fusion_branch": config.get("use_brand_fusion_branch", True),
        "use_category_fusion_branch": config.get("use_category_fusion_branch", True),
        "use_brand_score_branch": config.get("use_brand_score_branch", True),
        "use_category_score_branch": config.get("use_category_score_branch", True),
        "use_brand_profile": config.get("use_brand_profile", False),
        "use_category_profile": config.get("use_category_profile", False),
        "use_brand_memory": config.get("use_brand_memory", False),
        "use_category_memory": config.get("use_category_memory", False),
        "separate_sequence_fusion": config.get("separate_sequence_fusion", False),
        "hidden_size": config["hidden_size"],
        "num_heads": config["num_heads"],
        "num_blocks": config["num_blocks"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "lr_scheduler": config.get("lr_scheduler", "none"),
        "lr_scheduler_min_lr": config.get("lr_scheduler_min_lr", 0.0),
        "lr_scheduler_warmup_epochs": config.get("lr_scheduler_warmup_epochs", 0),
        "lr_scheduler_warmup_start_factor": config.get("lr_scheduler_warmup_start_factor", 0.2),
        "lr_scheduler_factor": config.get("lr_scheduler_factor", 0.5),
        "lr_scheduler_patience": config.get("lr_scheduler_patience", 1),
        "lr_scheduler_threshold": config.get("lr_scheduler_threshold", 0.0),
        "dropout": config["dropout"],
        "target_block_dim": config.get("target_block_dim"),
        "id_score_gate_init": config.get("id_score_gate_init"),
        "text_score_gate_init": config.get("text_score_gate_init"),
        "image_score_gate_init": config.get("image_score_gate_init"),
        "brand_score_gate_init": config.get("brand_score_gate_init"),
        "category_score_gate_init": config.get("category_score_gate_init"),
        "text_profile_gate_init": config.get("text_profile_gate_init"),
        "image_profile_gate_init": config.get("image_profile_gate_init"),
        "brand_profile_gate_init": config.get("brand_profile_gate_init"),
        "category_profile_gate_init": config.get("category_profile_gate_init"),
        "config": config,
    }


def load_result_summary(path: Path, *, name: str, family: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return summarize_result(name=name, family=family, result=data)


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "family",
        "experiment_name",
        "best_epoch",
        "stop_epoch",
        "best_val_ndcg10",
        "best_test_ndcg10",
        "best_test_hr10",
        "use_text_modality",
        "use_image_modality",
        "use_brand_branch",
        "use_category_branch",
        "use_brand_fusion_branch",
        "use_category_fusion_branch",
        "use_brand_score_branch",
        "use_category_score_branch",
        "use_brand_profile",
        "use_category_profile",
        "use_brand_memory",
        "use_category_memory",
        "separate_sequence_fusion",
        "hidden_size",
        "num_heads",
        "num_blocks",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "lr_scheduler",
        "lr_scheduler_min_lr",
        "lr_scheduler_warmup_epochs",
        "lr_scheduler_warmup_start_factor",
        "lr_scheduler_factor",
        "lr_scheduler_patience",
        "lr_scheduler_threshold",
        "dropout",
        "target_block_dim",
        "id_score_gate_init",
        "text_score_gate_init",
        "image_score_gate_init",
        "brand_score_gate_init",
        "category_score_gate_init",
        "text_profile_gate_init",
        "image_profile_gate_init",
        "brand_profile_gate_init",
        "category_profile_gate_init",
        "delta_vs_baseline",
        "relative_gain_vs_baseline_pct",
        "checkpoint_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Beauty2013 Paper Experiments",
        "",
        "| Variant | Family | Text | Image | SeqSplit | Cat Aux | Hidden | Heads | Blocks | Batch | LR | Scheduler | Scheduler Arg | Text Gate Init | Text Profile Init | Cat Profile Init | Image Gate Init | Best Epoch | Test NDCG@10 | Delta vs Baseline | Relative Gain |",
        "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        scheduler = row.get("lr_scheduler", "none")
        if scheduler == "cosine":
            scheduler_arg = "min_lr={min_lr}, warmup={warmup}".format(
                min_lr=row.get("lr_scheduler_min_lr"),
                warmup=row.get("lr_scheduler_warmup_epochs"),
            )
        elif scheduler == "plateau":
            scheduler_arg = "min_lr={min_lr}, factor={factor}, patience={patience}".format(
                min_lr=row.get("lr_scheduler_min_lr"),
                factor=row.get("lr_scheduler_factor"),
                patience=row.get("lr_scheduler_patience"),
            )
        else:
            scheduler_arg = "-"
        if row.get("use_category_branch", False):
            category_parts = []
            if row.get("use_category_fusion_branch", True):
                category_parts.append("fusion")
            if row.get("use_category_score_branch", True):
                category_parts.append("score")
            if row.get("use_category_profile", False):
                category_parts.append("profile")
            if row.get("use_category_memory", False):
                category_parts.append("memory")
            category_aux = "+".join(category_parts) if category_parts else "branch"
        elif row.get("use_category_memory", False):
            category_aux = "memory"
        else:
            category_aux = "-"
        lines.append(
            "| {variant} | {family} | {use_text_modality} | {use_image_modality} | {separate_sequence_fusion} | {category_aux} | {hidden_size} | {num_heads} | {num_blocks} | {batch_size} | {learning_rate} | {scheduler} | {scheduler_arg} | {text_score_gate_init} | {text_profile_gate_init} | {category_profile_gate_init} | {image_score_gate_init} | {best_epoch} | {best_test_ndcg10:.6f} | {delta_vs_baseline:.6f} | {relative_gain_vs_baseline_pct:.2f}% |".format(
                category_aux=category_aux,
                scheduler=scheduler,
                scheduler_arg=scheduler_arg,
                **row,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def write_latex(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Variant & Family & Hidden & Heads & Blocks & NDCG@10 & Gain (\%) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{variant} & {family} & {hidden} & {heads} & {blocks} & {ndcg:.6f} & {gain:.2f} \\\\".format(
                variant=_latex_escape(row["variant"]),
                family=_latex_escape(row["family"]),
                hidden=row["hidden_size"],
                heads=row["num_heads"],
                blocks=row["num_blocks"],
                ndcg=row["best_test_ndcg10"],
                gain=row["relative_gain_vs_baseline_pct"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_baseline_deltas(rows: list[dict]) -> list[dict]:
    baseline = next((row for row in rows if row["family"] == "baseline"), None)
    baseline_ndcg = None if baseline is None else baseline["best_test_ndcg10"]
    enriched = []
    for row in rows:
        row = dict(row)
        if baseline_ndcg is None or row["best_test_ndcg10"] is None:
            row["delta_vs_baseline"] = 0.0
            row["relative_gain_vs_baseline_pct"] = 0.0
        else:
            row["delta_vs_baseline"] = row["best_test_ndcg10"] - baseline_ndcg
            row["relative_gain_vs_baseline_pct"] = 0.0 if baseline_ndcg == 0 else (row["best_test_ndcg10"] / baseline_ndcg - 1.0) * 100.0
        enriched.append(row)
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Beauty2013 paper-style experiments.")
    parser.add_argument("--root-dir", default=ROOT_DIR)
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    parser.add_argument("--content-features-path", default=DEFAULT_CONTENT_FEATURES_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experiment-prefix", default="beauty2013_paper")
    parser.add_argument("--variants", default="baseline,mm_decomp_full_default")
    parser.add_argument("--summary-dir", default="outputs/paper/beauty2013")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variant_names = [name.strip() for name in args.variants.split(",") if name.strip()]
    summaries: list[dict] = []
    for name in variant_names:
        if name not in VARIANTS:
            raise KeyError(f"Unknown variant: {name}")
        if args.summarize_only:
            family = str(VARIANTS[name]["family"])
            result_path = Path(args.checkpoint_dir) / f"{args.experiment_prefix}_{name}.json"
            summaries.append(load_result_summary(result_path, name=name, family=family))
        else:
            summaries.append(
                run_variant(
                    name,
                    root_dir=args.root_dir,
                    checkpoint_dir=args.checkpoint_dir,
                    content_features_path=args.content_features_path,
                    device=args.device,
                    experiment_prefix=args.experiment_prefix,
                )
            )

    summaries = add_baseline_deltas(summaries)
    summary_dir = Path(args.summary_dir)
    write_csv(summaries, summary_dir / f"{args.experiment_prefix}_summary.csv")
    write_markdown(summaries, summary_dir / f"{args.experiment_prefix}_summary.md")
    write_latex(summaries, summary_dir / f"{args.experiment_prefix}_summary.tex")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
