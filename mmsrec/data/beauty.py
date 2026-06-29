from __future__ import annotations

import ast
import gzip
import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from mmsrec.utils import ensure_dir, write_json


def _stream_records(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                yield ast.literal_eval(line)


def _normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(_normalize_text(part) for part in value if part)
    if isinstance(value, dict):
        return " ".join(f"{key}: {_normalize_text(val)}" for key, val in value.items())
    return str(value).strip()


def _download(url: str, target: Path, timeout: int) -> None:
    if target.exists():
        return
    ensure_dir(target.parent)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; MMSRec/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)


def _sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _iterative_k_core(frame: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    current = frame
    while True:
        user_counts = current["reviewerID"].value_counts()
        item_counts = current["asin"].value_counts()
        next_frame = current[
            current["reviewerID"].isin(user_counts[user_counts >= min_user].index)
            & current["asin"].isin(item_counts[item_counts >= min_item].index)
        ]
        if len(next_frame) == len(current):
            return next_frame.reset_index(drop=True)
        current = next_frame


def _build_interactions(reviews_path: Path, cfg: "BeautyDataConfig") -> pd.DataFrame:
    rows = []
    for record in _stream_records(reviews_path):
        reviewer_id = record.get("reviewerID")
        asin = record.get("asin")
        timestamp = record.get("unixReviewTime")
        if not reviewer_id or not asin or timestamp is None:
            continue
        rows.append(
            {
                "reviewerID": reviewer_id,
                "asin": asin,
                "timestamp": int(timestamp),
                "overall": float(record.get("overall", 0.0)),
                "reviewTime": record.get("reviewTime", ""),
            }
        )

    interactions = pd.DataFrame(rows)
    interactions = interactions.sort_values(["reviewerID", "timestamp", "asin"]).reset_index(drop=True)
    interactions = _iterative_k_core(interactions, cfg.min_user_interactions, cfg.min_item_interactions)

    item_to_idx = {asin: idx for idx, asin in enumerate(sorted(interactions["asin"].unique()))}
    user_to_idx = {uid: idx for idx, uid in enumerate(sorted(interactions["reviewerID"].unique()))}
    interactions["item_id"] = interactions["asin"].map(item_to_idx)
    interactions["user_id"] = interactions["reviewerID"].map(user_to_idx)
    interactions = interactions.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    return interactions


def _build_items(metadata_path: Path, kept_asins: set[str]) -> pd.DataFrame:
    items = []
    for record in _stream_records(metadata_path):
        asin = record.get("asin")
        if asin not in kept_asins:
            continue
        title = _normalize_text(record.get("title"))
        description = _normalize_text(record.get("description"))
        brand = _normalize_text(record.get("brand"))
        categories = _normalize_text(record.get("categories"))
        image_url = record.get("imUrl") or ""
        text_input = " ".join(part for part in [title, description, categories, brand] if part).strip()
        if not text_input:
            text_input = f"asin {asin}"
        items.append(
            {
                "asin": asin,
                "title": title,
                "description": description,
                "brand": brand,
                "categories": categories,
                "imUrl": image_url,
                "text_input": text_input,
            }
        )

    frame = pd.DataFrame(items)
    existing = set(frame["asin"].tolist()) if not frame.empty else set()
    for asin in sorted(kept_asins - existing):
        frame = pd.concat(
            [
                frame,
                pd.DataFrame(
                    [
                        {
                            "asin": asin,
                            "title": "",
                            "description": "",
                            "brand": "",
                            "categories": "",
                            "imUrl": "",
                            "text_input": f"asin {asin}",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    item_to_idx = {asin: idx for idx, asin in enumerate(sorted(kept_asins))}
    frame["item_id"] = frame["asin"].map(item_to_idx)
    frame = frame.sort_values("item_id").reset_index(drop=True)
    return frame


def _pad_sequence(values: list[int], max_seq_len: int, pad_item_id: int) -> list[int]:
    if len(values) >= max_seq_len:
        return values[-max_seq_len:]
    return values + [pad_item_id] * (max_seq_len - len(values))


def _make_row(prefix: list[int], target: int, user_idx: int, max_seq_len: int, pad_item_id: int) -> dict:
    trimmed = prefix[-max_seq_len:]
    return {
        "user_id": user_idx,
        "seq": _pad_sequence(list(trimmed), max_seq_len, pad_item_id),
        "len_seq": max(len(trimmed), 1),
        "next": int(target),
    }


def _build_leave_one_out_sequences(interactions: pd.DataFrame, max_seq_len: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pad_item_id = int(interactions["item_id"].max()) + 1
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []

    for user_id, group in interactions.groupby("user_id", sort=True):
        items = group["item_id"].tolist()
        if len(items) < 3:
            continue
        train_items = items[:-2]
        val_item = items[-2]
        test_item = items[-1]
        if not train_items:
            continue
        for pos in range(1, len(train_items)):
            train_rows.append(_make_row(train_items[:pos], train_items[pos], int(user_id), max_seq_len, pad_item_id))
        val_rows.append(_make_row(train_items, val_item, int(user_id), max_seq_len, pad_item_id))
        test_rows.append(_make_row(items[:-1], test_item, int(user_id), max_seq_len, pad_item_id))

    return pd.DataFrame(train_rows), pd.DataFrame(val_rows), pd.DataFrame(test_rows)


@dataclass(slots=True)
class BeautyDataConfig:
    root_dir: str = "data/amazon_beauty_2013"
    raw_dir_name: str = "raw"
    artifacts_dir_name: str = "artifacts"
    max_seq_len: int = 50
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    reviews_url: str = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty.json.gz"
    metadata_url: str = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz"
    request_timeout: int = 30

    @property
    def root_path(self) -> Path:
        return Path(self.root_dir)

    @property
    def raw_path(self) -> Path:
        return self.root_path / self.raw_dir_name

    @property
    def artifacts_path(self) -> Path:
        return self.root_path / self.artifacts_dir_name


def build_dataset(cfg: BeautyDataConfig) -> dict:
    ensure_dir(cfg.raw_path)
    ensure_dir(cfg.artifacts_path)

    reviews_path = cfg.raw_path / Path(cfg.reviews_url).name
    metadata_path = cfg.raw_path / Path(cfg.metadata_url).name

    _download(cfg.reviews_url, reviews_path, timeout=cfg.request_timeout)
    _download(cfg.metadata_url, metadata_path, timeout=cfg.request_timeout)

    interactions = _build_interactions(reviews_path, cfg)
    items = _build_items(metadata_path, set(interactions["asin"].unique()))
    train_frame, val_frame, test_frame = _build_leave_one_out_sequences(interactions, cfg.max_seq_len)

    data_statis = pd.DataFrame(
        [
            {
                "seq_size": cfg.max_seq_len,
                "item_num": int(interactions["item_id"].nunique()),
                "user_num": int(interactions["user_id"].nunique()),
                "interaction_num": int(len(interactions)),
                "min_user_interactions": cfg.min_user_interactions,
                "min_item_interactions": cfg.min_item_interactions,
                "padding_item_id": int(interactions["item_id"].max()) + 1 if not interactions.empty else 0,
            }
        ]
    )

    items.to_pickle(cfg.artifacts_path / "items.df")
    interactions.to_pickle(cfg.artifacts_path / "interactions.df")
    train_frame.to_pickle(cfg.artifacts_path / "train_data.df")
    val_frame.to_pickle(cfg.artifacts_path / "val_data.df")
    test_frame.to_pickle(cfg.artifacts_path / "test_data.df")
    data_statis.to_pickle(cfg.artifacts_path / "data_statis.df")

    summary = {
        "raw_reviews_sha1": _sha1_file(reviews_path),
        "raw_metadata_sha1": _sha1_file(metadata_path),
        "user_count": int(interactions["user_id"].nunique()),
        "item_count": int(interactions["item_id"].nunique()),
        "interaction_count": int(len(interactions)),
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_frame)),
        "test_rows": int(len(test_frame)),
        "avg_train_seq_len": float(train_frame["len_seq"].mean()) if not train_frame.empty else 0.0,
    }
    write_json(cfg.artifacts_path / "dataset_summary.json", summary)
    return summary


def export_original_interactions_txt(root_dir: str, output_path: str) -> dict:
    root = Path(root_dir)
    artifacts = root / "artifacts"
    interactions = pd.read_pickle(artifacts / "interactions.df")
    target = Path(output_path)
    ensure_dir(target.parent)

    lines = [
        f"{int(row.user_id) + 1} {int(row.item_id) + 1}"
        for row in interactions.itertuples(index=False)
    ]
    target.write_text("\n".join(lines), encoding="utf-8")

    summary = json.loads((artifacts / "dataset_summary.json").read_text(encoding="utf-8"))
    manifest = {
        "source_root_dir": str(root),
        "source_artifacts_dir": str(artifacts),
        "output_path": str(target),
        "user_count": int(summary["user_count"]),
        "item_count": int(summary["item_count"]),
        "interaction_count": int(summary["interaction_count"]),
        "index_base": 1,
        "line_schema": "user_id item_id",
    }
    write_json(target.with_suffix(".manifest.json"), manifest)
    return manifest
