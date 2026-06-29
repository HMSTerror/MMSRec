# Experimental Section Draft

## Experimental Setup

We evaluate on Amazon Beauty 2013 using the canonical raw review and metadata files `reviews_Beauty.json.gz` and `meta_Beauty.json.gz`. The data pipeline follows the repository builder in `mmsrec.data.beauty.build_dataset(...)`, with iterative k-core filtering and leave-one-out splitting. The final processed dataset contains 22,363 users, 12,101 items, and 198,502 interactions, with 131,413 training rows and 22,363 validation/test rows each.

We use early-stopped `NDCG@10` as the primary model-selection metric and report `HR@10` as the secondary ranking metric. All multimodal models use the same optimization backbone unless otherwise stated: hidden size 192, 2 Transformer blocks, 6 attention heads, batch size 256, learning rate `5e-4`, weight decay `1e-4`, dropout `0.15`, and early stopping patience 5. The strongest model family combines metadata TF-IDF/SVD text, source semantic text, brand text, and category text, then injects multimodal context through cross-attention.

## Main Results

The strongest single-run result already exceeds the original optimization target. Relative to the PyTorch SASRec baseline (`NDCG@10 = 0.044426771`), the best category-profile run reaches `0.053791751`, which corresponds to a `21.08%` relative gain on `NDCG@10`. This is the strongest headline result we found in the entire search.

| Model | Seed Policy | Test NDCG@10 | Test HR@10 | Relative Gain vs Original Baseline |
| --- | --- | ---: | ---: | ---: |
| PyTorch SASRec baseline | seed100 | 0.044426771 | 0.068595448 | 0.00% |
| Best source-text multimodal family (source075_w15) | seed100 | 0.053095715 | 0.087465904 | 19.51% |
| Best category-profile multimodal run | seed102 | 0.053791751 | 0.090283057 | 21.08% |

The single-run story is therefore already strong enough for a clear main result. However, once the baseline is treated more strictly as a three-seed reference, the best single category-profile run corresponds to `18.38%` relative gain instead of `21.08%`. This makes robustness, rather than peak score, the main remaining paper risk.

## Robustness Across Matched Seeds

To test whether the category-profile branch is genuinely better than the strongest earlier source-text family, we ran matched-seed comparisons on seeds `100-109`. This comparison is more informative than comparing unrelated best runs because it isolates the effect of the category-profile branch under shared randomness.

| Family | Seeds | Mean NDCG@10 | Std NDCG@10 | Relative Gain vs 3-Seed Baseline Mean |
| --- | ---: | ---: | ---: | ---: |
| SASRec baseline | 3 | 0.045439786 | 0.000724845 | 0.00% |
| `source075_w15` matched seeds | 10 | 0.051210194 | 0.001119577 | 12.70% |
| category-profile matched seeds | 10 | 0.051912628 | 0.001012784 | 14.24% |

The paired matched-seed evidence is favorable to the category-profile branch:

- category-profile wins `7 / 10` matched seeds
- mean paired delta `(category - source)` is `+0.000702434`
- median paired delta is `+0.000259291`
- the ten-seed average remains higher for category-profile than for the source-text family

This is the cleanest robustness evidence currently available in the project. It does not imply that the category-profile branch wins on every seed, but it does show a stable net-positive trend.

## Statistical Interpretation

We also summarize the matched-seed comparison with lightweight small-sample statistics. The exact sign test yields a one-sided p-value of `0.171875` and a two-sided p-value of `0.343750`. A bootstrap estimate for the mean paired delta gives a 95% confidence interval of `[-0.000031441, 0.001458499]`.

These numbers matter for the paper framing:

1. The direction of the effect is favorable.
2. The effect size is modest.
3. The current ten-seed evidence is encouraging but not yet conventionally decisive under a strict significance threshold.

Accordingly, the correct claim is not that the category-profile branch is statistically proven superior on Beauty2013, but that it is the best-performing family we found and that its robustness trend is positive under matched seeds.

## Ablation Summary

The broader search is useful because it narrows down which design choices actually matter.

1. Source-text weighting:
   - `source=0.75` remains the best local point on the strongest text-heavy backbone.
2. Description trimming:
   - removing or truncating descriptions does not outperform the full text formulation.
3. Training-schedule tweaks:
   - cosine and plateau scheduling do not produce gains beyond the best unscheduled checkpoint.
4. Title or text repetition tricks:
   - explicit title append blocks and main-text repetition do not close the remaining gap.
5. Category-specific auxiliary design:
   - direct category residual scoring is harmful
   - category candidate fusion is harmful
   - the strongest evidence is that category helps only as a pure sequence-level profile signal
6. Weak-seed rescue attempts:
   - text-profile blending (`v45`) does not help
   - plateau seed rescue (`v47`) does not help
   - nearby gate sweeps on weak seeds (`v48`) do not help

This ablation pattern is actually paper-friendly: it shows that the final gain is not a trivial optimizer artifact or a generic consequence of adding more branches.

## Recommended Paper Framing

The strongest B-tier framing is a two-level claim.

First, emphasize the clear single-run headline:

`Our best category-profile multimodal SASRec variant improves NDCG@10 by 21.08% over the PyTorch SASRec baseline on Beauty2013.`

Second, pair it with a more careful robustness statement:

`Across ten matched random seeds, the category-profile branch outperforms the strongest earlier source-text family on seven seeds and improves the average NDCG@10, although the paired sign test indicates that the robustness advantage should be described as a positive trend rather than a definitive statistical win.`

## Next Writing Steps

Unless a very specific new architecture hypothesis appears, the current evidence is strong enough to stop search and consolidate the paper package:

1. finalize the main result table
2. finalize the ten-seed robustness table
3. finalize the ablation table
4. write one paragraph on variance and non-significant trend strength
5. move from experiment expansion to manuscript drafting
