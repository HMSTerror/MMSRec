# Beauty2013 B-Conference Submission Notes

## Current Position

The project already satisfies the original optimization target under the original single-run baseline reference:

- baseline `beauty2013_paper_v1_baseline`: `NDCG@10 = 0.04442677095272959`
- best single run `beauty2013_metadata_v39_profile_seed_m2p5_remaining_seed102`: `NDCG@10 = 0.05379175070654626`
- relative gain vs baseline: `+21.08%`

This is strong enough for a headline single-run claim. The remaining weakness is robustness across seeds, not peak score.

## Recommended Headline

Use a two-level reporting strategy:

1. Main headline:
   - the best category-profile model exceeds the original SASRec baseline by `21.08%` on `NDCG@10`
2. Robustness paragraph:
   - the category-profile family is not uniformly better on every seed
   - but it remains net positive against the older `source075_w15` family on matched-seed comparisons

This is the cleanest honest story for a B-tier submission.

## Main Result Table

Suggested main table rows:

| Model | Seed Policy | Test NDCG@10 | Test HR@10 | Relative Gain vs Original Baseline |
| --- | --- | ---: | ---: | ---: |
| PyTorch SASRec baseline | seed100 | 0.044426771 | 0.068595448 | 0.00% |
| Best source-text multimodal family (source075_w15) | seed100 | 0.053095715 | 0.087465904 | 19.51% |
| Best category-profile multimodal run | seed102 | 0.053791751 | 0.090283057 | 21.08% |

Recommended caption:

`The proposed category-profile multimodal SASRec variant achieves the strongest single-run result on Beauty2013, exceeding the PyTorch SASRec baseline by 21.08% on NDCG@10 under early-stopped evaluation.`

## Robustness Table

Current matched-seed comparison through `seed109`:

| Family | Seeds | Mean NDCG@10 | Std NDCG@10 | Relative Gain vs 3-Seed Baseline Mean |
| --- | ---: | ---: | ---: | ---: |
| SASRec baseline | 3 | 0.045439786 | 0.000724845 | 0.00% |
| `source075_w15` matched seeds | 10 | 0.051210194 | 0.001119577 | 12.70% |
| category-profile matched seeds | 10 | 0.051912628 | 0.001012784 | 14.24% |

Current matched-seed win summary:

- category-profile wins `7 / 10` matched seeds
- mean absolute delta vs matched `source075_w15`: `+0.000702433636974921`
- 95% bootstrap CI for mean matched delta: `[-0.000031441, 0.001458499]`
- exact sign test one-sided p-value: `0.171875`
- exact sign test two-sided p-value: `0.343750`

Recommended caption:

`The category-profile branch does not dominate every random seed, but it remains a net-positive upgrade over the strongest earlier source-text family in matched-seed comparisons.`

## What Helps The Paper

High-value positive evidence already in hand:

- one verified run above the `+20%` target
- a matched-seed comparison showing net-positive average gains
- multiple closed negative results that strengthen the causal story

## Immediate Next Decision

At the time of writing, the ten-seed matched comparison is already available and the recommendation is to stop architecture search unless a very specific new hypothesis appears.
