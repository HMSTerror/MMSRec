from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mmsrec.utils import ensure_dir


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text if text else ""


def _build_text_embeddings(
    texts: list[str],
    *,
    text_svd_dim: int,
    min_df: int,
    max_text_features: int | None,
    ngram_max: int,
    random_state: int,
    stop_words: str | None = "english",
) -> np.ndarray:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import Normalizer

    documents = [text if text else "unknown item" for text in texts]
    vectorizer_params = {
        "lowercase": True,
        "stop_words": stop_words,
        "min_df": min_df,
        "max_features": max_text_features,
        "ngram_range": (1, max(1, ngram_max)),
    }
    try:
        tfidf = TfidfVectorizer(**vectorizer_params).fit_transform(documents)
    except ValueError:
        fallback_params = dict(vectorizer_params)
        fallback_params["min_df"] = 1
        fallback_params["stop_words"] = None
        tfidf = TfidfVectorizer(**fallback_params).fit_transform(documents)
    if tfidf.shape[1] == 0:
        raise ValueError("TF-IDF vocabulary is empty; cannot build metadata text embeddings.")

    max_rank = min(tfidf.shape[0] - 1, tfidf.shape[1] - 1)
    if max_rank <= 0:
        dense = tfidf.toarray().astype(np.float32)
        return _pad_embedding_dim(dense, text_svd_dim)

    n_components = min(text_svd_dim, max_rank)
    reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
    reduced = reducer.fit_transform(tfidf).astype(np.float32)
    Normalizer(copy=False).fit_transform(reduced)
    return _pad_embedding_dim(reduced, text_svd_dim)


def _pad_embedding_dim(array: np.ndarray, target_dim: int) -> np.ndarray:
    if array.shape[1] >= target_dim:
        return array
    padding = np.zeros((array.shape[0], target_dim - array.shape[1]), dtype=array.dtype)
    return np.concatenate([array, padding], axis=1)


def _compose_weighted_text_inputs(
    items: pd.DataFrame,
    *,
    extra_title_repeats: int = 0,
    extra_category_repeats: int = 0,
    extra_brand_repeats: int = 0,
) -> list[str]:
    texts = [_safe_text(value) for value in items["text_input"].tolist()]
    titles = [_safe_text(value) for value in items.get("title", pd.Series([""] * len(items))).tolist()]
    categories = [_safe_text(value) for value in items.get("categories", pd.Series([""] * len(items))).tolist()]
    brands = [_safe_text(value) for value in items.get("brand", pd.Series([""] * len(items))).tolist()]

    weighted_texts = []
    for title, category, brand, text in zip(titles, categories, brands, texts, strict=True):
        parts = []
        if title and extra_title_repeats > 0:
            parts.extend([title] * extra_title_repeats)
        if category and extra_category_repeats > 0:
            parts.extend([category] * extra_category_repeats)
        if brand and extra_brand_repeats > 0:
            parts.extend([brand] * extra_brand_repeats)
        if text:
            parts.append(text)
        weighted_texts.append(" ".join(parts).strip() if parts else "unknown item")
    return weighted_texts


def build_metadata_content_features(
    *,
    items_df_path: str,
    source_content_features_path: str,
    output_path: str,
    text_svd_dim: int = 256,
    min_df: int = 2,
    max_text_features: int | None = 50000,
    ngram_max: int = 2,
    random_state: int = 100,
    append_source_text_embeddings: bool = False,
    append_title_embeddings: bool = False,
    append_brand_embeddings: bool = False,
    append_category_embeddings: bool = False,
    extra_title_repeats_in_text: int = 0,
    extra_category_repeats_in_text: int = 0,
    extra_brand_repeats_in_text: int = 0,
    title_svd_dim: int = 64,
    brand_svd_dim: int = 32,
    category_svd_dim: int = 64,
    source_text_weight: float = 1.0,
    title_weight: float = 1.0,
    brand_weight: float = 1.0,
    category_weight: float = 1.0,
    concat_brand_into_text_embeddings: bool = True,
    concat_category_into_text_embeddings: bool = True,
    store_separate_structured_embeddings: bool = False,
) -> dict:
    import torch

    items = pd.read_pickle(items_df_path).sort_values("item_id").reset_index(drop=True)
    if items.empty:
        raise ValueError("items.df is empty; cannot build metadata content features.")

    texts = _compose_weighted_text_inputs(
        items,
        extra_title_repeats=extra_title_repeats_in_text,
        extra_category_repeats=extra_category_repeats_in_text,
        extra_brand_repeats=extra_brand_repeats_in_text,
    )
    text_embeddings = _build_text_embeddings(
        texts,
        text_svd_dim=text_svd_dim,
        min_df=min_df,
        max_text_features=max_text_features,
        ngram_max=ngram_max,
        random_state=random_state,
    )

    source = torch.load(source_content_features_path, map_location="cpu")
    source_asins = list(source.get("asins", []))
    source_text_embeddings = source["text_embeddings"].float()
    source_text_available = source["text_available"].to(dtype=torch.int64)
    source_image_embeddings = source["image_embeddings"].float()
    source_image_available = source["image_available"].to(dtype=torch.int64)
    if not source_asins:
        raise ValueError("source content features do not contain ASINs for alignment.")

    asin_to_index = {asin: idx for idx, asin in enumerate(source_asins)}
    item_asins = items["asin"].tolist()
    text_dim = int(source_text_embeddings.shape[1])
    image_dim = int(source_image_embeddings.shape[1])
    aligned_source_text_embeddings = torch.zeros((len(item_asins), text_dim), dtype=torch.float32)
    aligned_image_embeddings = torch.zeros((len(item_asins), image_dim), dtype=torch.float32)
    aligned_image_available = torch.zeros((len(item_asins),), dtype=torch.int64)

    missing_image_count = 0
    for row_idx, asin in enumerate(item_asins):
        source_idx = asin_to_index.get(asin)
        if source_idx is None:
            missing_image_count += 1
            continue
        if int(source_text_available[source_idx].item()) == 1:
            aligned_source_text_embeddings[row_idx] = source_text_embeddings[source_idx]
        aligned_image_embeddings[row_idx] = source_image_embeddings[source_idx]
        aligned_image_available[row_idx] = source_image_available[source_idx]
        if int(source_image_available[source_idx].item()) == 0:
            missing_image_count += 1

    if append_source_text_embeddings:
        source_text_block = aligned_source_text_embeddings.numpy() * float(source_text_weight)
        text_embeddings = np.concatenate(
            [text_embeddings, source_text_block],
            axis=1,
        ).astype(np.float32)

    if append_title_embeddings:
        title_texts = [_safe_text(value) for value in items.get("title", pd.Series([""] * len(items))).tolist()]
        title_embeddings = _build_text_embeddings(
            title_texts,
            text_svd_dim=title_svd_dim,
            min_df=1,
            max_text_features=max_text_features,
            ngram_max=2,
            random_state=random_state,
            stop_words=None,
        )
        title_embeddings = title_embeddings * float(title_weight)
        text_embeddings = np.concatenate([text_embeddings, title_embeddings], axis=1).astype(np.float32)

    brand_available = None
    category_available = None
    brand_payload_embeddings = None
    category_payload_embeddings = None

    if append_brand_embeddings:
        brand_texts = [_safe_text(value) for value in items.get("brand", pd.Series([""] * len(items))).tolist()]
        brand_available = torch.tensor([1 if text else 0 for text in brand_texts], dtype=torch.int64)
        brand_embeddings = _build_text_embeddings(
            brand_texts,
            text_svd_dim=brand_svd_dim,
            min_df=1,
            max_text_features=max_text_features,
            ngram_max=1,
            random_state=random_state,
            stop_words=None,
        )
        brand_embeddings = brand_embeddings * float(brand_weight)
        if concat_brand_into_text_embeddings:
            text_embeddings = np.concatenate([text_embeddings, brand_embeddings], axis=1).astype(np.float32)
        if store_separate_structured_embeddings:
            brand_payload_embeddings = brand_embeddings * brand_available.unsqueeze(1).numpy().astype(np.float32)

    if append_category_embeddings:
        category_texts = [_safe_text(value) for value in items.get("categories", pd.Series([""] * len(items))).tolist()]
        category_available = torch.tensor([1 if text else 0 for text in category_texts], dtype=torch.int64)
        category_embeddings = _build_text_embeddings(
            category_texts,
            text_svd_dim=category_svd_dim,
            min_df=1,
            max_text_features=max_text_features,
            ngram_max=2,
            random_state=random_state,
            stop_words=None,
        )
        category_embeddings = category_embeddings * float(category_weight)
        if concat_category_into_text_embeddings:
            text_embeddings = np.concatenate([text_embeddings, category_embeddings], axis=1).astype(np.float32)
        if store_separate_structured_embeddings:
            category_payload_embeddings = category_embeddings * category_available.unsqueeze(1).numpy().astype(np.float32)

    payload = {
        "item_ids": torch.arange(len(item_asins), dtype=torch.long),
        "text_embeddings": torch.from_numpy(text_embeddings).to(dtype=torch.float32),
        "image_embeddings": aligned_image_embeddings,
        "text_available": torch.ones((len(item_asins),), dtype=torch.int64),
        "image_available": aligned_image_available,
        "asins": item_asins,
    }
    if brand_payload_embeddings is not None and brand_available is not None:
        payload["brand_embeddings"] = torch.from_numpy(brand_payload_embeddings).to(dtype=torch.float32)
        payload["brand_available"] = brand_available
    if category_payload_embeddings is not None and category_available is not None:
        payload["category_embeddings"] = torch.from_numpy(category_payload_embeddings).to(dtype=torch.float32)
        payload["category_available"] = category_available

    target = Path(output_path)
    ensure_dir(target.parent)
    torch.save(payload, target)

    summary = {
        "item_count": len(item_asins),
        "text_embedding_dim": int(payload["text_embeddings"].shape[1]),
        "image_embedding_dim": int(payload["image_embeddings"].shape[1]),
        "text_available_count": int(payload["text_available"].sum().item()),
        "image_available_count": int(payload["image_available"].sum().item()),
        "missing_image_count": int(missing_image_count),
        "output_path": str(target),
    }
    if "brand_available" in payload:
        summary["brand_available_count"] = int(payload["brand_available"].sum().item())
    if "category_available" in payload:
        summary["category_available_count"] = int(payload["category_available"].sum().item())
    return summary
