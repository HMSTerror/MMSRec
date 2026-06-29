from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from mmsrec.utils import ensure_dir


@dataclass(slots=True)
class BaselineTrainConfig:
    root_dir: str
    checkpoint_dir: str = "outputs/checkpoints"
    experiment_name: str = "sasrec_baseline"
    device: str = "cuda"
    epochs: int = 50
    batch_size: int = 256
    eval_batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    random_seed: int = 100
    hidden_size: int = 64
    num_heads: int = 1
    num_blocks: int = 1
    dropout: float = 0.1
    topk: tuple[int, ...] = (10, 20, 50)
    early_stop_metric: str = "NDCG@10"
    early_stop_patience: int | None = 5
    early_stop_min_delta: float = 0.0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_artifacts(root_dir: str) -> dict:
    artifacts_dir = Path(root_dir) / "artifacts"
    train_data = pd.read_pickle(artifacts_dir / "train_data.df").reset_index(drop=True)
    val_data = pd.read_pickle(artifacts_dir / "val_data.df").reset_index(drop=True)
    test_data = pd.read_pickle(artifacts_dir / "test_data.df").reset_index(drop=True)
    statis = pd.read_pickle(artifacts_dir / "data_statis.df").iloc[0]
    return {
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "item_num": int(statis["item_num"]),
        "user_num": int(statis["user_num"]),
        "interaction_num": int(statis["interaction_num"]),
        "seq_size": int(statis["seq_size"]),
        "padding_item_id": int(statis.get("padding_item_id", statis["item_num"])),
    }


def _batch_indices(size: int, batch_size: int, *, shuffle: bool, rng: np.random.Generator) -> list[np.ndarray]:
    indices = np.arange(size, dtype=np.int64)
    if shuffle:
        rng.shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, size, batch_size)]


def _tensor_batch(frame: pd.DataFrame, indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    batch = frame.iloc[indices.tolist()]
    seq = torch.tensor(batch["seq"].tolist(), dtype=torch.long, device=device)
    lengths = torch.tensor(batch["len_seq"].tolist(), dtype=torch.long, device=device)
    targets = torch.tensor(batch["next"].tolist(), dtype=torch.long, device=device)
    return {"seq": seq, "len_seq": lengths, "target": targets}


class SASRecBaseline(nn.Module):
    def __init__(
        self,
        *,
        item_num: int,
        seq_size: int,
        hidden_size: int,
        num_heads: int,
        num_blocks: int,
        dropout: float,
        padding_item_id: int,
    ) -> None:
        super().__init__()
        self.item_num = item_num
        self.seq_size = seq_size
        self.hidden_size = hidden_size
        self.padding_item_id = padding_item_id

        self.item_embeddings = nn.Embedding(item_num + 1, hidden_size, padding_idx=padding_item_id)
        self.position_embeddings = nn.Embedding(seq_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.output = nn.Linear(hidden_size, item_num)

        nn.init.normal_(self.item_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        if self.item_embeddings.padding_idx is not None:
            with torch.no_grad():
                self.item_embeddings.weight[self.item_embeddings.padding_idx].zero_()

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = seq.shape
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.item_embeddings(seq) + self.position_embeddings(positions)
        hidden = self.embedding_dropout(hidden)

        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=seq.device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = seq.eq(self.padding_item_id)
        encoded = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)

        last_index = lengths.clamp(min=1, max=seq_len) - 1
        state = encoded[torch.arange(batch_size, device=seq.device), last_index]
        return self.output(state)


def _apply_history_mask(scores: torch.Tensor, seq: torch.Tensor, targets: torch.Tensor, padding_item_id: int) -> torch.Tensor:
    masked = scores.clone()
    seq_cpu = seq.detach().cpu().numpy()
    targets_cpu = targets.detach().cpu().numpy()
    for row_idx, history in enumerate(seq_cpu):
        target = int(targets_cpu[row_idx])
        for item_id in history.tolist():
            item_id = int(item_id)
            if item_id == padding_item_id or item_id == target:
                continue
            masked[row_idx, item_id] = -torch.inf
    return masked


def _compute_metrics(scores: torch.Tensor, targets: torch.Tensor, topk: tuple[int, ...]) -> dict[str, float]:
    max_k = max(topk)
    top_indices = torch.topk(scores, k=max_k, dim=1).indices
    metrics = {f"HR@{k}": 0.0 for k in topk}
    metrics.update({f"NDCG@{k}": 0.0 for k in topk})

    for row_idx, target in enumerate(targets.tolist()):
        ranked = top_indices[row_idx].tolist()
        for k in topk:
            window = ranked[:k]
            if target in window:
                rank = window.index(target)
                metrics[f"HR@{k}"] += 1.0
                metrics[f"NDCG@{k}"] += 1.0 / math.log2(rank + 2.0)

    count = max(int(targets.shape[0]), 1)
    for key in list(metrics.keys()):
        metrics[key] /= count
    metrics["sample_count"] = int(targets.shape[0])
    return metrics


def _evaluate(
    model: SASRecBaseline,
    dataset: pd.DataFrame,
    *,
    device: torch.device,
    batch_size: int,
    topk: tuple[int, ...],
    padding_item_id: int,
) -> dict[str, float]:
    model.eval()
    all_scores = []
    all_targets = []
    with torch.no_grad():
        for indices in _batch_indices(len(dataset), batch_size, shuffle=False, rng=np.random.default_rng(0)):
            batch = _tensor_batch(dataset, indices, device)
            logits = model(batch["seq"], batch["len_seq"])
            logits = _apply_history_mask(logits, batch["seq"], batch["target"], padding_item_id)
            all_scores.append(logits.cpu())
            all_targets.append(batch["target"].cpu())

    if not all_scores:
        return _compute_metrics(torch.zeros((0, model.item_num)), torch.zeros((0,), dtype=torch.long), topk)

    merged_scores = torch.cat(all_scores, dim=0)
    merged_targets = torch.cat(all_targets, dim=0)
    return _compute_metrics(merged_scores, merged_targets, topk)


def train_sasrec_baseline(config: BaselineTrainConfig) -> dict:
    _seed_everything(config.random_seed)
    artifacts = _load_artifacts(config.root_dir)
    requested_device = torch.device(config.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = requested_device

    model = SASRecBaseline(
        item_num=artifacts["item_num"],
        seq_size=artifacts["seq_size"],
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
        padding_item_id=artifacts["padding_item_id"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(config.random_seed)

    checkpoint_dir = ensure_dir(config.checkpoint_dir)
    checkpoint_path = checkpoint_dir / f"{config.experiment_name}.pt"
    report_path = checkpoint_dir / f"{config.experiment_name}.json"

    history: dict[str, list] = {"train_loss": [], "val_metrics": [], "test_metrics": []}
    best_metric = float("-inf")
    best_epoch = 0
    best_val_metrics = None
    best_test_metrics = None
    best_state = None
    patience_left = config.early_stop_patience
    stop_epoch = config.epochs
    stopped_early = False

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []
        for indices in _batch_indices(len(artifacts["train_data"]), config.batch_size, shuffle=True, rng=rng):
            batch = _tensor_batch(artifacts["train_data"], indices, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["seq"], batch["len_seq"])
            loss = criterion(logits, batch["target"])
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        history["train_loss"].append(mean_loss)

        val_metrics = _evaluate(
            model,
            artifacts["val_data"],
            device=device,
            batch_size=config.eval_batch_size,
            topk=config.topk,
            padding_item_id=artifacts["padding_item_id"],
        )
        val_metrics["epoch"] = epoch
        history["val_metrics"].append(val_metrics)

        test_metrics = _evaluate(
            model,
            artifacts["test_data"],
            device=device,
            batch_size=config.eval_batch_size,
            topk=config.topk,
            padding_item_id=artifacts["padding_item_id"],
        )
        test_metrics["epoch"] = epoch
        history["test_metrics"].append(test_metrics)

        current_metric = float(val_metrics[config.early_stop_metric])
        improved = current_metric > best_metric + config.early_stop_min_delta
        if improved:
            best_metric = current_metric
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_test_metrics = dict(test_metrics)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = config.early_stop_patience
        elif patience_left is not None:
            patience_left -= 1
            if patience_left <= 0:
                stop_epoch = epoch
                stopped_early = True
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save({"model_state_dict": best_state, "config": asdict(config)}, checkpoint_path)
    else:
        torch.save({"model_state_dict": model.state_dict(), "config": asdict(config)}, checkpoint_path)

    result = {
        "checkpoint_path": str(checkpoint_path),
        "history": {
            **history,
            "best_epoch": best_epoch,
            "best_metric_name": config.early_stop_metric,
            "best_metric_value": best_metric if best_metric != float("-inf") else None,
            "best_val_metrics": best_val_metrics,
            "best_test_metrics": best_test_metrics,
            "stop_epoch": stop_epoch,
            "stopped_early": stopped_early,
            "early_stop_patience": config.early_stop_patience,
            "early_stop_min_delta": config.early_stop_min_delta,
        },
        "config": asdict(config),
    }
    report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
