# MMSRec

MMSRec is a multimodal sequential recommendation workspace built on top of the original SASRec repository.

The repository now has two layers:

1. Legacy SASRec reference code in the project root (`main.py`, `model.py`, `modules.py`, `sampler.py`, `util.py`), kept for compatibility with the original TensorFlow 1 / Python 2 implementation.
2. A new `mmsrec` Python package that adds a reproducible Amazon Beauty 2013 data pipeline, a PyTorch SASRec baseline, and a multimodal SASRec variant that injects text and image features through cross-attention.

## What is in this repo

- `mmsrec/data/beauty.py`
  - Builds Beauty 2013 artifacts directly from raw `reviews_Beauty.json.gz` and `meta_Beauty.json.gz`.
  - Applies iterative user/item k-core filtering.
  - Produces `items.df`, `interactions.df`, train/val/test leave-one-out splits, and `dataset_summary.json`.
  - Exports the original SASRec text format (`user_id item_id`, 1-based indexing) for baseline comparisons.

- `mmsrec/baselines/sasrec.py`
  - Pure PyTorch SASRec baseline trainer.
  - Supports early stopping and checkpoint export.
  - Persists per-epoch validation/test metrics plus the best validation and best test snapshot in the JSON report.

- `mmsrec/multimodal/sasrec.py`
  - Loads external content features from `content_features.pt`.
  - Aligns text/image embeddings to Beauty item order.
  - Encodes behavior sequences with a Transformer encoder.
  - Builds modality memory tokens from text and image features.
  - Injects multimodal context into sequence states through cross-attention.
  - Scores candidate items with ID, text, image, and availability-aware fused representations.

- `tests/`
  - Unit tests for the Beauty pipeline, baseline trainer, multimodal feature loading, model forward pass, and multimodal smoke training.

## Repository layout

```text
.
|-- data/
|   |-- Beauty.txt
|   |-- DataProcessing.py
|   |-- Steam.txt
|   `-- ml-1m.txt
|-- mmsrec/
|   |-- baselines/
|   |-- data/
|   `-- multimodal/
|-- tests/
|-- main.py
|-- model.py
|-- modules.py
|-- sampler.py
`-- util.py
```

## Data pipeline

The current workflow starts from raw Amazon Beauty files:

- `reviews_Beauty.json.gz`
- `meta_Beauty.json.gz`

`build_dataset(...)` writes canonical artifacts under:

```text
<root>/artifacts/
  items.df
  interactions.df
  train_data.df
  val_data.df
  test_data.df
  data_statis.df
  dataset_summary.json
```

`export_original_interactions_txt(...)` then produces the original SASRec-compatible interaction file:

```text
<root>/exports/Beauty.txt
<root>/exports/Beauty.manifest.json
```

## Training flow

Recommended flow for Beauty 2013:

1. Build dataset artifacts from raw files.
2. Export `Beauty.txt` for parity with the original SASRec setup.
3. Train the PyTorch SASRec baseline.
4. Train the multimodal SASRec model with external text/image embeddings.
5. Compare early-stopped best-checkpoint metrics, not just the last epoch.

## Example usage

Build Beauty 2013 artifacts:

```python
from mmsrec.data.beauty import BeautyDataConfig, build_dataset, export_original_interactions_txt

cfg = BeautyDataConfig(root_dir="data/amazon_beauty_2013")
build_dataset(cfg)
export_original_interactions_txt(cfg.root_dir, "data/amazon_beauty_2013/exports/Beauty.txt")
```

Train the PyTorch SASRec baseline:

```python
from mmsrec.baselines.sasrec import BaselineTrainConfig, train_sasrec_baseline

result = train_sasrec_baseline(
    BaselineTrainConfig(
        root_dir="data/amazon_beauty_2013",
        checkpoint_dir="outputs/checkpoints",
        experiment_name="beauty2013_sasrec",
        device="cuda",
    )
)
```

Train the multimodal SASRec model:

```python
from mmsrec.multimodal.sasrec import MMTrainConfig, train_mm_sasrec

result = train_mm_sasrec(
    MMTrainConfig(
        root_dir="data/amazon_beauty_2013",
        content_features_path="data/amazon_beauty_2013/artifacts/content_features.pt",
        checkpoint_dir="outputs/checkpoints",
        experiment_name="beauty2013_mm_sasrec",
        device="cuda",
    )
)
```

Both trainers write:

- a checkpoint `.pt`
- a JSON report with per-epoch training history
- best checkpoint metadata, including `best_epoch`, `best_val_metrics`, and `best_test_metrics`

## Tests

Run the focused test suite with:

```powershell
py -3 -m unittest tests.test_beauty_pipeline
py -3 -m unittest tests.test_sasrec_baseline
py -3 -m unittest tests.test_mm_sasrec
py -3 -m unittest tests.test_mm_sasrec_training
```

If PyTorch is not installed in the local interpreter, the training-related tests will skip.

## Notes

- The legacy TensorFlow 1 code is preserved, but the actively developed path is the PyTorch-based `mmsrec` package.
- The repository expects external multimodal features to be prepared as a `content_features.pt` payload containing text/image embeddings, availability masks, item IDs, and optional ASINs.
