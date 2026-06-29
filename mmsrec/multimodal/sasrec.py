from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from mmsrec.baselines.sasrec import (
    _apply_history_mask,
    _batch_indices,
    _compute_metrics,
    _load_artifacts,
    _seed_everything,
    _tensor_batch,
)
from mmsrec.utils import ensure_dir


def load_content_features(path: str, items_df_path: str | None = None) -> dict[str, torch.Tensor | list[str]]:
    payload = torch.load(path, map_location="cpu")
    text_embeddings = payload["text_embeddings"].float()
    image_embeddings = payload["image_embeddings"].float()
    text_available = payload["text_available"].to(dtype=torch.int64)
    image_available = payload["image_available"].to(dtype=torch.int64)
    brand_embeddings = payload.get("brand_embeddings")
    category_embeddings = payload.get("category_embeddings")
    brand_available = payload.get("brand_available")
    category_available = payload.get("category_available")
    if brand_embeddings is not None:
        brand_embeddings = brand_embeddings.float()
    if category_embeddings is not None:
        category_embeddings = category_embeddings.float()
    if brand_available is not None:
        brand_available = brand_available.to(dtype=torch.int64)
    if category_available is not None:
        category_available = category_available.to(dtype=torch.int64)
    asins = list(payload.get("asins", []))
    item_ids = payload["item_ids"].to(dtype=torch.long)

    if items_df_path is not None and asins:
        items = pd.read_pickle(items_df_path).sort_values("item_id").reset_index(drop=True)
        asin_to_index = {asin: idx for idx, asin in enumerate(asins)}
        reorder = []
        for asin in items["asin"].tolist():
            reorder.append(asin_to_index.get(asin, -1))

        def _align_matrix(matrix: torch.Tensor | None) -> torch.Tensor | None:
            if matrix is None:
                return None
            aligned = torch.zeros((len(reorder), matrix.shape[1]), dtype=matrix.dtype)
            for target_idx, source_idx in enumerate(reorder):
                if source_idx < 0:
                    continue
                aligned[target_idx] = matrix[source_idx]
            return aligned

        def _align_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
            if mask is None:
                return None
            aligned = torch.zeros((len(reorder),), dtype=mask.dtype)
            for target_idx, source_idx in enumerate(reorder):
                if source_idx < 0:
                    continue
                aligned[target_idx] = mask[source_idx]
            return aligned

        aligned_text = torch.zeros((len(reorder), text_embeddings.shape[1]), dtype=text_embeddings.dtype)
        aligned_image = torch.zeros((len(reorder), image_embeddings.shape[1]), dtype=image_embeddings.dtype)
        aligned_text_available = torch.zeros((len(reorder),), dtype=text_available.dtype)
        aligned_image_available = torch.zeros((len(reorder),), dtype=image_available.dtype)
        for target_idx, source_idx in enumerate(reorder):
            if source_idx < 0:
                continue
            aligned_text[target_idx] = text_embeddings[source_idx]
            aligned_image[target_idx] = image_embeddings[source_idx]
            aligned_text_available[target_idx] = text_available[source_idx]
            aligned_image_available[target_idx] = image_available[source_idx]

        text_embeddings = aligned_text
        image_embeddings = aligned_image
        text_available = aligned_text_available
        image_available = aligned_image_available
        brand_embeddings = _align_matrix(brand_embeddings)
        category_embeddings = _align_matrix(category_embeddings)
        brand_available = _align_mask(brand_available)
        category_available = _align_mask(category_available)
        item_ids = torch.arange(len(reorder), dtype=torch.long)
        asins = items["asin"].tolist()

    pad_text = torch.zeros((1, text_embeddings.shape[1]), dtype=text_embeddings.dtype)
    pad_image = torch.zeros((1, image_embeddings.shape[1]), dtype=image_embeddings.dtype)
    pad_mask = torch.zeros((1,), dtype=text_available.dtype)

    result: dict[str, torch.Tensor | list[str]] = {
        "item_ids": torch.cat([item_ids, torch.tensor([int(item_ids.numel())], dtype=item_ids.dtype)], dim=0),
        "text_embeddings": torch.cat([text_embeddings, pad_text], dim=0),
        "image_embeddings": torch.cat([image_embeddings, pad_image], dim=0),
        "text_available": torch.cat([text_available, pad_mask], dim=0),
        "image_available": torch.cat([image_available, pad_mask.clone()], dim=0),
        "asins": asins,
    }
    if brand_embeddings is not None:
        result["brand_embeddings"] = torch.cat(
            [brand_embeddings, torch.zeros((1, brand_embeddings.shape[1]), dtype=brand_embeddings.dtype)],
            dim=0,
        )
    if category_embeddings is not None:
        result["category_embeddings"] = torch.cat(
            [category_embeddings, torch.zeros((1, category_embeddings.shape[1]), dtype=category_embeddings.dtype)],
            dim=0,
        )
    if brand_available is not None:
        result["brand_available"] = torch.cat([brand_available, pad_mask.clone()], dim=0)
    if category_available is not None:
        result["category_available"] = torch.cat([category_available, pad_mask.clone()], dim=0)
    return result


def _set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def _resolve_epoch_learning_rate(
    *,
    epoch: int,
    total_epochs: int,
    base_learning_rate: float,
    scheduler_name: str,
    scheduler_min_lr: float,
    scheduler_warmup_epochs: int,
    scheduler_warmup_start_factor: float,
) -> float:
    if epoch < 1:
        raise ValueError(f"epoch must be >= 1, got {epoch}")
    if total_epochs < 1:
        raise ValueError(f"total_epochs must be >= 1, got {total_epochs}")
    if epoch > total_epochs:
        raise ValueError(f"epoch {epoch} exceeds total_epochs {total_epochs}")
    if base_learning_rate <= 0.0:
        raise ValueError(f"base_learning_rate must be positive, got {base_learning_rate}")
    if scheduler_min_lr < 0.0:
        raise ValueError(f"scheduler_min_lr must be >= 0, got {scheduler_min_lr}")
    if scheduler_min_lr > base_learning_rate:
        raise ValueError(
            f"scheduler_min_lr must be <= base_learning_rate, got {scheduler_min_lr} > {base_learning_rate}"
        )
    if scheduler_warmup_epochs < 0:
        raise ValueError(f"scheduler_warmup_epochs must be >= 0, got {scheduler_warmup_epochs}")
    if not 0.0 <= scheduler_warmup_start_factor <= 1.0:
        raise ValueError(
            "scheduler_warmup_start_factor must be between 0 and 1 inclusive, "
            f"got {scheduler_warmup_start_factor}"
        )

    normalized_name = scheduler_name.strip().lower()
    if normalized_name in {"", "none"}:
        return float(base_learning_rate)
    if normalized_name != "cosine":
        raise ValueError(f"unsupported scheduler_name: {scheduler_name}")
    if scheduler_warmup_epochs >= total_epochs:
        raise ValueError(
            "scheduler_warmup_epochs must be smaller than total_epochs for cosine scheduling, "
            f"got {scheduler_warmup_epochs} and {total_epochs}"
        )

    if scheduler_warmup_epochs > 0 and epoch <= scheduler_warmup_epochs:
        warmup_progress = epoch / scheduler_warmup_epochs
        warmup_factor = scheduler_warmup_start_factor + (1.0 - scheduler_warmup_start_factor) * warmup_progress
        return float(base_learning_rate * warmup_factor)

    cosine_epochs = max(total_epochs - scheduler_warmup_epochs, 1)
    cosine_step = epoch - scheduler_warmup_epochs - 1
    cosine_progress = 0.0 if cosine_epochs <= 1 else min(max(cosine_step / (cosine_epochs - 1), 0.0), 1.0)
    cosine_weight = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
    return float(scheduler_min_lr + (base_learning_rate - scheduler_min_lr) * cosine_weight)


def _step_plateau_learning_rate(
    *,
    current_learning_rate: float,
    current_metric: float,
    best_metric: float,
    bad_epoch_count: int,
    factor: float,
    patience: int,
    min_learning_rate: float,
    threshold: float,
) -> tuple[float, float, int]:
    if current_learning_rate <= 0.0:
        raise ValueError(f"current_learning_rate must be positive, got {current_learning_rate}")
    if factor <= 0.0 or factor >= 1.0:
        raise ValueError(f"factor must be between 0 and 1, got {factor}")
    if patience < 0:
        raise ValueError(f"patience must be >= 0, got {patience}")
    if min_learning_rate < 0.0:
        raise ValueError(f"min_learning_rate must be >= 0, got {min_learning_rate}")
    if threshold < 0.0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")

    improved = current_metric > best_metric + threshold
    if improved:
        return float(current_learning_rate), float(current_metric), 0

    next_bad_epoch_count = bad_epoch_count + 1
    if next_bad_epoch_count <= patience:
        return float(current_learning_rate), float(best_metric), next_bad_epoch_count

    next_learning_rate = max(float(min_learning_rate), float(current_learning_rate) * float(factor))
    return float(next_learning_rate), float(best_metric), 0

class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm_2 = nn.LayerNorm(hidden_size)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor) -> torch.Tensor:
        if not bool(memory_mask.any().item()):
            return query
        row_has_memory = memory_mask.any(dim=1)
        if bool(row_has_memory.all().item()):
            attn_output, _ = self.cross_attn(
                query=query,
                key=memory,
                value=memory,
                key_padding_mask=~memory_mask,
                need_weights=False,
            )
            hidden = self.norm_1(query + self.dropout(attn_output))
            ff_output = self.ff(hidden)
            return self.norm_2(hidden + self.dropout(ff_output))

        output = query.clone()
        valid_rows = row_has_memory.nonzero(as_tuple=False).flatten()
        valid_query = query.index_select(0, valid_rows)
        valid_memory = memory.index_select(0, valid_rows)
        valid_mask = memory_mask.index_select(0, valid_rows)
        attn_output, _ = self.cross_attn(
            query=valid_query,
            key=valid_memory,
            value=valid_memory,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        hidden = self.norm_1(valid_query + self.dropout(attn_output))
        ff_output = self.ff(hidden)
        valid_output = self.norm_2(hidden + self.dropout(ff_output))
        output.index_copy_(0, valid_rows, valid_output)
        return output


class MMSASRec(nn.Module):
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
        text_dim: int,
        image_dim: int,
        brand_dim: int | None = None,
        category_dim: int | None = None,
        target_block_dim: int,
        use_text_modality: bool = True,
        use_image_modality: bool = True,
        use_brand_branch: bool = False,
        use_category_branch: bool = False,
        use_brand_fusion_branch: bool = True,
        use_category_fusion_branch: bool = True,
        use_brand_score_branch: bool = True,
        use_category_score_branch: bool = True,
        use_brand_profile: bool = False,
        use_category_profile: bool = False,
        use_brand_memory: bool = False,
        use_category_memory: bool = False,
        separate_sequence_fusion: bool = False,
        id_score_gate_init: float = -2.0,
        text_score_gate_init: float = -2.0,
        image_score_gate_init: float = -2.0,
        brand_score_gate_init: float = -2.0,
        category_score_gate_init: float = -2.0,
        text_profile_gate_init: float = -1.0,
        image_profile_gate_init: float = -3.0,
        brand_profile_gate_init: float = -3.0,
        category_profile_gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.item_num = item_num
        self.seq_size = seq_size
        self.hidden_size = hidden_size
        self.padding_item_id = padding_item_id
        self.target_block_dim = target_block_dim
        self.use_text_modality = use_text_modality
        self.use_image_modality = use_image_modality
        self.use_brand_branch = use_brand_branch and brand_dim is not None
        self.use_category_branch = use_category_branch and category_dim is not None
        self.use_brand_fusion_branch = self.use_brand_branch and use_brand_fusion_branch
        self.use_category_fusion_branch = self.use_category_branch and use_category_fusion_branch
        self.use_brand_score_branch = self.use_brand_branch and use_brand_score_branch
        self.use_category_score_branch = self.use_category_branch and use_category_score_branch
        self.use_brand_profile = self.use_brand_branch and use_brand_profile
        self.use_category_profile = self.use_category_branch and use_category_profile
        self.use_brand_memory = use_brand_memory and brand_dim is not None
        self.use_category_memory = use_category_memory and category_dim is not None
        self.separate_sequence_fusion = separate_sequence_fusion
        if self.separate_sequence_fusion and (self.use_brand_memory or self.use_category_memory):
            raise ValueError("structured memory branches are not supported with separate_sequence_fusion=True")

        next_modality_type_id = 2
        self.brand_memory_modality_id: int | None = None
        if self.use_brand_memory:
            self.brand_memory_modality_id = next_modality_type_id
            next_modality_type_id += 1
        self.category_memory_modality_id: int | None = None
        if self.use_category_memory:
            self.category_memory_modality_id = next_modality_type_id
            next_modality_type_id += 1

        self.item_embeddings = nn.Embedding(item_num + 1, hidden_size, padding_idx=padding_item_id)
        self.position_embeddings = nn.Embedding(seq_size, hidden_size)
        self.modality_type_embeddings = nn.Embedding(next_modality_type_id, hidden_size)
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
        self.cross_block = CrossAttentionBlock(hidden_size, num_heads, dropout)
        self.text_cross_block = CrossAttentionBlock(hidden_size, num_heads, dropout)
        self.image_cross_block = CrossAttentionBlock(hidden_size, num_heads, dropout)

        self.text_memory_projection = nn.Linear(text_dim, hidden_size)
        self.image_memory_projection = nn.Linear(image_dim, hidden_size)
        self.brand_memory_projection = (
            nn.Linear(brand_dim, hidden_size) if self.use_brand_memory and brand_dim is not None else None
        )
        self.category_memory_projection = (
            nn.Linear(category_dim, hidden_size) if self.use_category_memory and category_dim is not None else None
        )

        # Explicit target space: z_txt in R^d, z_img in R^d, with decomposed residual scoring.
        self.text_target_projection = nn.Linear(text_dim, target_block_dim)
        self.image_target_projection = nn.Linear(image_dim, target_block_dim)
        self.brand_target_projection = (
            nn.Linear(brand_dim, target_block_dim) if self.use_brand_branch and brand_dim is not None else None
        )
        self.category_target_projection = (
            nn.Linear(category_dim, target_block_dim) if self.use_category_branch and category_dim is not None else None
        )
        structured_branch_count = int(self.use_brand_fusion_branch) + int(self.use_category_fusion_branch)
        fused_input_dim = hidden_size + (2 + structured_branch_count) * target_block_dim + 2 + structured_branch_count
        self.candidate_fusion = nn.Sequential(
            nn.Linear(fused_input_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

        self.id_query_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.text_query_projection = nn.Sequential(
            nn.Linear(hidden_size, target_block_dim),
            nn.GELU(),
            nn.LayerNorm(target_block_dim),
        )
        self.image_query_projection = nn.Sequential(
            nn.Linear(hidden_size, target_block_dim),
            nn.GELU(),
            nn.LayerNorm(target_block_dim),
        )
        self.brand_query_projection = (
            nn.Sequential(
                nn.Linear(hidden_size, target_block_dim),
                nn.GELU(),
                nn.LayerNorm(target_block_dim),
            )
            if self.use_brand_branch
            else None
        )
        self.category_query_projection = (
            nn.Sequential(
                nn.Linear(hidden_size, target_block_dim),
                nn.GELU(),
                nn.LayerNorm(target_block_dim),
            )
            if self.use_category_branch
            else None
        )
        self.sequence_fusion_gate = nn.Sequential(
            nn.Linear(hidden_size * 3 + 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )
        self.sequence_fusion_ffn = nn.Sequential(
            nn.Linear(hidden_size * 3 + 2, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.sequence_fusion_norm = nn.LayerNorm(hidden_size)

        self.id_score_gate = nn.Parameter(torch.tensor(float(id_score_gate_init)))
        self.text_score_gate = nn.Parameter(torch.tensor(float(text_score_gate_init)))
        self.image_score_gate = nn.Parameter(torch.tensor(float(image_score_gate_init)))
        self.brand_score_gate = nn.Parameter(torch.tensor(float(brand_score_gate_init)))
        self.category_score_gate = nn.Parameter(torch.tensor(float(category_score_gate_init)))
        self.text_profile_gate = nn.Parameter(torch.tensor(float(text_profile_gate_init)))
        self.image_profile_gate = nn.Parameter(torch.tensor(float(image_profile_gate_init)))
        self.brand_profile_gate = nn.Parameter(torch.tensor(float(brand_profile_gate_init)))
        self.category_profile_gate = nn.Parameter(torch.tensor(float(category_profile_gate_init)))

        nn.init.normal_(self.item_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.modality_type_embeddings.weight, mean=0.0, std=0.02)
        if self.item_embeddings.padding_idx is not None:
            with torch.no_grad():
                self.item_embeddings.weight[self.item_embeddings.padding_idx].zero_()

    def _build_sequence_hidden(self, seq: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = seq.shape
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.item_embeddings(seq) + self.position_embeddings(positions)
        return self.embedding_dropout(hidden)

    def _encode_behavior(self, hidden: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        seq_len = seq.shape[1]
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=seq.device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = seq.eq(self.padding_item_id)
        return self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)

    def _build_memory(
        self,
        seq: torch.Tensor,
        content_embeddings: torch.Tensor,
        content_available: torch.Tensor,
        projection: nn.Module,
        modality_type_id: int,
        enabled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = seq.shape
        positions = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        seq_flat = seq.view(-1)

        content_seq = content_embeddings.index_select(0, seq_flat).view(batch_size, seq_len, -1)
        content_mask = content_available.index_select(0, seq_flat).view(batch_size, seq_len).bool()
        padding_mask = seq.ne(self.padding_item_id)
        content_mask = content_mask & padding_mask
        if not enabled:
            content_mask = torch.zeros_like(content_mask)

        base_positions = self.position_embeddings(positions)
        content_tokens = (
            projection(content_seq)
            + base_positions
            + self.modality_type_embeddings(torch.full_like(positions, modality_type_id))
        )
        return content_tokens, content_mask

    def _candidate_embeddings(
        self,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        text_available: torch.Tensor,
        image_available: torch.Tensor,
        brand_embeddings: torch.Tensor | None,
        category_embeddings: torch.Tensor | None,
        brand_available: torch.Tensor | None,
        category_available: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        item_ids = torch.arange(self.item_num, device=text_embeddings.device)
        id_repr = self.item_embeddings(item_ids)
        text_gate = text_available[: self.item_num].unsqueeze(1).to(id_repr.dtype)
        image_gate = image_available[: self.item_num].unsqueeze(1).to(id_repr.dtype)
        if not self.use_text_modality:
            text_gate = torch.zeros_like(text_gate)
        if not self.use_image_modality:
            image_gate = torch.zeros_like(image_gate)

        text_repr = self.text_target_projection(text_embeddings[: self.item_num]) * text_gate
        image_repr = self.image_target_projection(image_embeddings[: self.item_num]) * image_gate
        branch_features = [id_repr, text_repr, image_repr]
        gate_features = [text_gate, image_gate]

        brand_repr = None
        if self.use_brand_branch and self.brand_target_projection is not None:
            if brand_embeddings is None or brand_available is None:
                brand_gate = torch.zeros((self.item_num, 1), dtype=id_repr.dtype, device=id_repr.device)
                brand_repr = torch.zeros((self.item_num, self.target_block_dim), dtype=id_repr.dtype, device=id_repr.device)
            else:
                brand_gate = brand_available[: self.item_num].unsqueeze(1).to(id_repr.dtype)
                brand_repr = self.brand_target_projection(brand_embeddings[: self.item_num]) * brand_gate
            if self.use_brand_fusion_branch:
                branch_features.append(brand_repr)
                gate_features.append(brand_gate)

        category_repr = None
        if self.use_category_branch and self.category_target_projection is not None:
            if category_embeddings is None or category_available is None:
                category_gate = torch.zeros((self.item_num, 1), dtype=id_repr.dtype, device=id_repr.device)
                category_repr = torch.zeros((self.item_num, self.target_block_dim), dtype=id_repr.dtype, device=id_repr.device)
            else:
                category_gate = category_available[: self.item_num].unsqueeze(1).to(id_repr.dtype)
                category_repr = self.category_target_projection(category_embeddings[: self.item_num]) * category_gate
            if self.use_category_fusion_branch:
                branch_features.append(category_repr)
                gate_features.append(category_gate)

        fused = torch.cat(branch_features + gate_features, dim=1)
        fused_repr = self.candidate_fusion(fused) + id_repr
        return id_repr, text_repr, image_repr, brand_repr, category_repr, fused_repr

    def _build_sequence_profile(
        self,
        seq: torch.Tensor,
        content_embeddings: torch.Tensor,
        content_available: torch.Tensor,
        projection: nn.Module,
        *,
        enabled: bool,
    ) -> torch.Tensor:
        batch_size, seq_len = seq.shape
        if not enabled:
            output_dim = projection.out_features
            return torch.zeros((batch_size, output_dim), device=seq.device, dtype=content_embeddings.dtype)

        seq_flat = seq.view(-1)
        content_seq = content_embeddings.index_select(0, seq_flat).view(batch_size, seq_len, -1)
        content_proj = projection(content_seq)
        content_mask = content_available.index_select(0, seq_flat).view(batch_size, seq_len).bool()
        content_mask = content_mask & seq.ne(self.padding_item_id)
        recency = torch.arange(1, seq_len + 1, device=seq.device, dtype=content_proj.dtype).view(1, seq_len, 1)
        weights = content_mask.unsqueeze(-1).to(content_proj.dtype) * recency
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (content_proj * weights).sum(dim=1) / denom

    def forward(
        self,
        *,
        seq: torch.Tensor,
        lengths: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
        brand_embeddings: torch.Tensor | None = None,
        category_embeddings: torch.Tensor | None = None,
        text_available: torch.Tensor,
        image_available: torch.Tensor,
        brand_available: torch.Tensor | None = None,
        category_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self._build_sequence_hidden(seq)
        encoded = self._encode_behavior(hidden, seq)

        text_memory, text_mask = self._build_memory(
            seq,
            text_embeddings,
            text_available,
            self.text_memory_projection,
            modality_type_id=0,
            enabled=self.use_text_modality,
        )
        image_memory, image_mask = self._build_memory(
            seq,
            image_embeddings,
            image_available,
            self.image_memory_projection,
            modality_type_id=1,
            enabled=self.use_image_modality,
        )
        brand_memory = None
        brand_mask = None
        if self.use_brand_memory:
            if brand_embeddings is None or brand_available is None or self.brand_memory_projection is None:
                raise ValueError("brand memory is enabled but brand embeddings are unavailable")
            if self.brand_memory_modality_id is None:
                raise ValueError("brand memory modality id is not initialized")
            brand_memory, brand_mask = self._build_memory(
                seq,
                brand_embeddings,
                brand_available,
                self.brand_memory_projection,
                modality_type_id=self.brand_memory_modality_id,
                enabled=True,
            )
        category_memory = None
        category_mask = None
        if self.use_category_memory:
            if category_embeddings is None or category_available is None or self.category_memory_projection is None:
                raise ValueError("category memory is enabled but category embeddings are unavailable")
            if self.category_memory_modality_id is None:
                raise ValueError("category memory modality id is not initialized")
            category_memory, category_mask = self._build_memory(
                seq,
                category_embeddings,
                category_available,
                self.category_memory_projection,
                modality_type_id=self.category_memory_modality_id,
                enabled=True,
            )

        if self.separate_sequence_fusion:
            text_enriched = self.text_cross_block(encoded, text_memory, text_mask)
            image_enriched = self.image_cross_block(encoded, image_memory, image_mask)
            text_presence = text_mask.unsqueeze(-1).to(encoded.dtype)
            image_presence = image_mask.unsqueeze(-1).to(encoded.dtype)
            fusion_input = torch.cat([encoded, text_enriched, image_enriched, text_presence, image_presence], dim=-1)
            fusion_gates = torch.sigmoid(self.sequence_fusion_gate(fusion_input))
            enriched = encoded
            enriched = enriched + fusion_gates[..., 0:1] * (text_enriched - encoded)
            enriched = enriched + fusion_gates[..., 1:2] * (image_enriched - encoded)
            enriched = self.sequence_fusion_norm(enriched + self.sequence_fusion_ffn(fusion_input))
        else:
            memories = [text_memory, image_memory]
            masks = [text_mask, image_mask]
            if brand_memory is not None and brand_mask is not None:
                memories.append(brand_memory)
                masks.append(brand_mask)
            if category_memory is not None and category_mask is not None:
                memories.append(category_memory)
                masks.append(category_mask)
            memory = torch.cat(memories, dim=1)
            memory_mask = torch.cat(masks, dim=1)
            enriched = self.cross_block(encoded, memory, memory_mask)

        last_index = lengths.clamp(min=1, max=seq.shape[1]) - 1
        batch_index = torch.arange(seq.shape[0], device=seq.device)
        state = enriched[batch_index, last_index]

        id_repr, text_repr, image_repr, brand_repr, category_repr, fused_repr = self._candidate_embeddings(
            text_embeddings,
            image_embeddings,
            text_available,
            image_available,
            brand_embeddings,
            category_embeddings,
            brand_available,
            category_available,
        )
        text_profile = self._build_sequence_profile(
            seq,
            text_embeddings,
            text_available,
            self.text_target_projection,
            enabled=self.use_text_modality,
        )
        image_profile = self._build_sequence_profile(
            seq,
            image_embeddings,
            image_available,
            self.image_target_projection,
            enabled=self.use_image_modality,
        )
        brand_profile = None
        if self.use_brand_branch and self.brand_target_projection is not None and brand_embeddings is not None and brand_available is not None:
            brand_profile = self._build_sequence_profile(
                seq,
                brand_embeddings,
                brand_available,
                self.brand_target_projection,
                enabled=self.use_brand_profile,
            )
        category_profile = None
        if (
            self.use_category_branch
            and self.category_target_projection is not None
            and category_embeddings is not None
            and category_available is not None
        ):
            category_profile = self._build_sequence_profile(
                seq,
                category_embeddings,
                category_available,
                self.category_target_projection,
                enabled=self.use_category_profile,
            )

        scores = state @ fused_repr.t()
        scores = scores + torch.sigmoid(self.id_score_gate) * (self.id_query_projection(state) @ id_repr.t())
        if self.use_text_modality:
            scores = scores + torch.sigmoid(self.text_score_gate) * (self.text_query_projection(state) @ text_repr.t())
            scores = scores + torch.sigmoid(self.text_profile_gate) * (text_profile @ text_repr.t())
        if self.use_image_modality:
            scores = scores + torch.sigmoid(self.image_score_gate) * (self.image_query_projection(state) @ image_repr.t())
            scores = scores + torch.sigmoid(self.image_profile_gate) * (image_profile @ image_repr.t())
        if self.use_brand_score_branch and brand_repr is not None and self.brand_query_projection is not None:
            scores = scores + torch.sigmoid(self.brand_score_gate) * (self.brand_query_projection(state) @ brand_repr.t())
        if self.use_category_score_branch and category_repr is not None and self.category_query_projection is not None:
            scores = scores + torch.sigmoid(self.category_score_gate) * (self.category_query_projection(state) @ category_repr.t())
        if self.use_brand_profile and brand_repr is not None and brand_profile is not None:
            scores = scores + torch.sigmoid(self.brand_profile_gate) * (brand_profile @ brand_repr.t())
        if self.use_category_profile and category_repr is not None and category_profile is not None:
            scores = scores + torch.sigmoid(self.category_profile_gate) * (category_profile @ category_repr.t())
        return scores


@dataclass(slots=True)
class MMTrainConfig:
    root_dir: str
    content_features_path: str
    checkpoint_dir: str = "outputs/checkpoints"
    experiment_name: str = "mm_sasrec"
    device: str = "cuda"
    epochs: int = 50
    batch_size: int = 256
    eval_batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    lr_scheduler: str = "none"
    lr_scheduler_min_lr: float = 0.0
    lr_scheduler_warmup_epochs: int = 0
    lr_scheduler_warmup_start_factor: float = 0.2
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 1
    lr_scheduler_threshold: float = 0.0
    random_seed: int = 100
    hidden_size: int = 128
    num_heads: int = 4
    num_blocks: int = 2
    dropout: float = 0.1
    topk: tuple[int, ...] = (10, 20, 50)
    early_stop_metric: str = "NDCG@10"
    early_stop_patience: int | None = 5
    early_stop_min_delta: float = 0.0
    target_block_dim: int = 128
    use_text_modality: bool = True
    use_image_modality: bool = True
    use_brand_branch: bool = False
    use_category_branch: bool = False
    use_brand_fusion_branch: bool = True
    use_category_fusion_branch: bool = True
    use_brand_score_branch: bool = True
    use_category_score_branch: bool = True
    use_brand_profile: bool = False
    use_category_profile: bool = False
    use_brand_memory: bool = False
    use_category_memory: bool = False
    separate_sequence_fusion: bool = False
    id_score_gate_init: float = -2.0
    text_score_gate_init: float = -2.0
    image_score_gate_init: float = -2.0
    brand_score_gate_init: float = -2.0
    category_score_gate_init: float = -2.0
    text_profile_gate_init: float = -1.0
    image_profile_gate_init: float = -3.0
    brand_profile_gate_init: float = -3.0
    category_profile_gate_init: float = -2.0


def _load_mm_artifacts(config: MMTrainConfig) -> dict:
    base = _load_artifacts(config.root_dir)
    content = load_content_features(
        config.content_features_path,
        items_df_path=str(Path(config.root_dir) / "artifacts" / "items.df"),
    )
    base.update(content)
    return base


def _evaluate_mm(
    model: MMSASRec,
    dataset: pd.DataFrame,
    *,
    device: torch.device,
    batch_size: int,
    topk: tuple[int, ...],
    padding_item_id: int,
    text_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    text_available: torch.Tensor,
    image_available: torch.Tensor,
    brand_embeddings: torch.Tensor | None,
    category_embeddings: torch.Tensor | None,
    brand_available: torch.Tensor | None,
    category_available: torch.Tensor | None,
) -> dict[str, float]:
    model.eval()
    all_scores = []
    all_targets = []
    with torch.no_grad():
        for indices in _batch_indices(len(dataset), batch_size, shuffle=False, rng=np.random.default_rng(0)):
            batch = _tensor_batch(dataset, indices, device)
            logits = model(
                seq=batch["seq"],
                lengths=batch["len_seq"],
                text_embeddings=text_embeddings,
                image_embeddings=image_embeddings,
                brand_embeddings=brand_embeddings,
                category_embeddings=category_embeddings,
                text_available=text_available,
                image_available=image_available,
                brand_available=brand_available,
                category_available=category_available,
            )
            logits = _apply_history_mask(logits, batch["seq"], batch["target"], padding_item_id)
            all_scores.append(logits.cpu())
            all_targets.append(batch["target"].cpu())
    if not all_scores:
        return _compute_metrics(torch.zeros((0, model.item_num)), torch.zeros((0,), dtype=torch.long), topk)
    merged_scores = torch.cat(all_scores, dim=0)
    merged_targets = torch.cat(all_targets, dim=0)
    return _compute_metrics(merged_scores, merged_targets, topk)


def train_mm_sasrec(config: MMTrainConfig) -> dict:
    _seed_everything(config.random_seed)
    artifacts = _load_mm_artifacts(config)
    requested_device = torch.device(config.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = requested_device

    model = MMSASRec(
        item_num=artifacts["item_num"],
        seq_size=artifacts["seq_size"],
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
        padding_item_id=artifacts["padding_item_id"],
        text_dim=int(artifacts["text_embeddings"].shape[1]),
        image_dim=int(artifacts["image_embeddings"].shape[1]),
        brand_dim=None if "brand_embeddings" not in artifacts else int(artifacts["brand_embeddings"].shape[1]),
        category_dim=None if "category_embeddings" not in artifacts else int(artifacts["category_embeddings"].shape[1]),
        target_block_dim=config.target_block_dim,
        use_text_modality=config.use_text_modality,
        use_image_modality=config.use_image_modality,
        use_brand_branch=config.use_brand_branch,
        use_category_branch=config.use_category_branch,
        use_brand_fusion_branch=config.use_brand_fusion_branch,
        use_category_fusion_branch=config.use_category_fusion_branch,
        use_brand_score_branch=config.use_brand_score_branch,
        use_category_score_branch=config.use_category_score_branch,
        use_brand_profile=config.use_brand_profile,
        use_category_profile=config.use_category_profile,
        use_brand_memory=config.use_brand_memory,
        use_category_memory=config.use_category_memory,
        separate_sequence_fusion=config.separate_sequence_fusion,
        id_score_gate_init=config.id_score_gate_init,
        text_score_gate_init=config.text_score_gate_init,
        image_score_gate_init=config.image_score_gate_init,
        brand_score_gate_init=config.brand_score_gate_init,
        category_score_gate_init=config.category_score_gate_init,
        text_profile_gate_init=config.text_profile_gate_init,
        image_profile_gate_init=config.image_profile_gate_init,
        brand_profile_gate_init=config.brand_profile_gate_init,
        category_profile_gate_init=config.category_profile_gate_init,
    ).to(device)

    text_embeddings = artifacts["text_embeddings"].to(device)
    image_embeddings = artifacts["image_embeddings"].to(device)
    text_available = artifacts["text_available"].to(device)
    image_available = artifacts["image_available"].to(device)
    brand_embeddings = None if "brand_embeddings" not in artifacts else artifacts["brand_embeddings"].to(device)
    category_embeddings = None if "category_embeddings" not in artifacts else artifacts["category_embeddings"].to(device)
    brand_available = None if "brand_available" not in artifacts else artifacts["brand_available"].to(device)
    category_available = None if "category_available" not in artifacts else artifacts["category_available"].to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    rng = np.random.default_rng(config.random_seed)

    checkpoint_dir = ensure_dir(config.checkpoint_dir)
    checkpoint_path = checkpoint_dir / f"{config.experiment_name}.pt"
    report_path = checkpoint_dir / f"{config.experiment_name}.json"

    history: dict[str, list] = {"train_loss": [], "val_metrics": [], "test_metrics": [], "learning_rates": []}
    best_metric = float("-inf")
    best_epoch = 0
    best_val_metrics = None
    best_test_metrics = None
    best_state = None
    patience_left = config.early_stop_patience
    stop_epoch = config.epochs
    stopped_early = False
    plateau_learning_rate = float(config.learning_rate)
    plateau_best_metric = float("-inf")
    plateau_bad_epoch_count = 0

    for epoch in range(1, config.epochs + 1):
        if config.lr_scheduler.strip().lower() == "plateau":
            epoch_learning_rate = float(plateau_learning_rate)
        else:
            epoch_learning_rate = _resolve_epoch_learning_rate(
                epoch=epoch,
                total_epochs=config.epochs,
                base_learning_rate=config.learning_rate,
                scheduler_name=config.lr_scheduler,
                scheduler_min_lr=config.lr_scheduler_min_lr,
                scheduler_warmup_epochs=config.lr_scheduler_warmup_epochs,
                scheduler_warmup_start_factor=config.lr_scheduler_warmup_start_factor,
            )
        _set_optimizer_learning_rate(optimizer, epoch_learning_rate)
        history["learning_rates"].append(float(epoch_learning_rate))

        model.train()
        epoch_losses = []
        for indices in _batch_indices(len(artifacts["train_data"]), config.batch_size, shuffle=True, rng=rng):
            batch = _tensor_batch(artifacts["train_data"], indices, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                seq=batch["seq"],
                lengths=batch["len_seq"],
                text_embeddings=text_embeddings,
                image_embeddings=image_embeddings,
                brand_embeddings=brand_embeddings,
                category_embeddings=category_embeddings,
                text_available=text_available,
                image_available=image_available,
                brand_available=brand_available,
                category_available=category_available,
            )
            loss = criterion(logits, batch["target"])
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        mean_loss = float(sum(epoch_losses) / len(epoch_losses)) if epoch_losses else 0.0
        history["train_loss"].append(mean_loss)

        val_metrics = _evaluate_mm(
            model,
            artifacts["val_data"],
            device=device,
            batch_size=config.eval_batch_size,
            topk=config.topk,
            padding_item_id=artifacts["padding_item_id"],
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            text_available=text_available,
            image_available=image_available,
            brand_embeddings=brand_embeddings,
            category_embeddings=category_embeddings,
            brand_available=brand_available,
            category_available=category_available,
        )
        val_metrics["epoch"] = epoch
        history["val_metrics"].append(val_metrics)

        test_metrics = _evaluate_mm(
            model,
            artifacts["test_data"],
            device=device,
            batch_size=config.eval_batch_size,
            topk=config.topk,
            padding_item_id=artifacts["padding_item_id"],
            text_embeddings=text_embeddings,
            image_embeddings=image_embeddings,
            text_available=text_available,
            image_available=image_available,
            brand_embeddings=brand_embeddings,
            category_embeddings=category_embeddings,
            brand_available=brand_available,
            category_available=category_available,
        )
        test_metrics["epoch"] = epoch
        history["test_metrics"].append(test_metrics)

        current_metric = float(val_metrics[config.early_stop_metric])
        if config.lr_scheduler.strip().lower() == "plateau":
            plateau_learning_rate, plateau_best_metric, plateau_bad_epoch_count = _step_plateau_learning_rate(
                current_learning_rate=plateau_learning_rate,
                current_metric=current_metric,
                best_metric=plateau_best_metric,
                bad_epoch_count=plateau_bad_epoch_count,
                factor=config.lr_scheduler_factor,
                patience=config.lr_scheduler_patience,
                min_learning_rate=config.lr_scheduler_min_lr,
                threshold=config.lr_scheduler_threshold,
            )
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
