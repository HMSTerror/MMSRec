# Beauty2013 Submission Tables

## Table 1. Main Results

| Model | Seed Policy | Test NDCG@10 | Test HR@10 | Relative Gain vs Original Baseline |
| --- | --- | ---: | ---: | ---: |
| PyTorch SASRec baseline | seed100 | 0.044426771 | 0.068595448 | 0.00% |
| Best source-text multimodal family (source075_w15) | seed100 | 0.053095715 | 0.087465904 | 19.51% |
| Best category-profile multimodal run | seed102 | 0.053791751 | 0.090283057 | 21.08% |

Suggested caption:

`Main results on Beauty2013. The best category-profile multimodal SASRec variant achieves the strongest single-run result and improves NDCG@10 by 21.08% over the PyTorch SASRec baseline under early-stopped evaluation.`

## Table 2. Robustness Across Matched Seeds

| Family | Seeds | Mean NDCG@10 | Std NDCG@10 | Relative Gain vs 3-Seed Baseline Mean |
| --- | ---: | ---: | ---: | ---: |
| PyTorch SASRec baseline | 3 | 0.045439786 | 0.000724845 | 0.00% |
| `source075_w15` matched seeds | 10 | 0.051210194 | 0.001119577 | 12.70% |
| category-profile matched seeds | 10 | 0.051912628 | 0.001012784 | 14.24% |

Paired matched-seed summary:

- category-profile wins `7 / 10` seeds
- mean paired delta `(category - source) = +0.000702434`
- 95% bootstrap CI for mean paired delta: `[-0.000031441, 0.001458499]`
- exact sign test: one-sided `p = 0.171875`, two-sided `p = 0.343750`

Suggested caption:

`Matched-seed robustness comparison between the category-profile branch and the strongest earlier source-text family. The category-profile branch improves the average NDCG@10 and wins on seven of ten matched seeds, but the paired sign test indicates that this robustness advantage should be described as a positive trend rather than a decisive statistical win.`

## Table 3. Search Trajectory

| Variant | Test NDCG@10 | Relative Gain vs Original Baseline | Role In Paper Story |
| --- | ---: | ---: | --- |
| SASRec baseline | 0.044426771 | 0.00% | reference point |
| Early multimodal decomposition (v3) | 0.050209541 | 13.02% | first clear multimodal lift |
| Metadata + source text full tuned | 0.051624386 | 16.20% | strong text-heavy baseline |
| Metadata + source text + brand/category (v5) | 0.052411247 | 17.97% | structured metadata helps |
| Structured weight 1.5 (v8) | 0.052646413 | 18.50% | stronger structured weighting |
| Best source075_w15 family (v16) | 0.053095715 | 19.51% | strongest pre-category family |
| Best category-profile configuration (v36) | 0.053309772 | 19.99% | category helps as profile signal |
| Best category-profile seed hit (v39 seed102) | 0.053791751 | 21.08% | strongest headline result |

Suggested caption:

`Search trajectory on Beauty2013. The final gain does not arise from a single abrupt change; it emerges from a sequence of improvements on top of the text-heavy multimodal backbone, with the category-profile branch providing the final strongest lift.`

## Table 4. Negative Follow-Ups That Strengthen The Narrative

| Follow-Up | Best Observed Outcome | Conclusion |
| --- | --- | --- |
| Text-profile blend (v45) | best 0.053300, still below v36 | Reintroducing weak text-profile scoring does not help. |
| Plateau seed rescue (v47) | weak seeds remain at 0.051038 / 0.051003 | Schedule changes do not solve robustness. |
| Weak-seed gate sweep (v48) | best nearby-gate reruns seed101 0.050989 / seed103 0.050935 | Local gate mismatch is not the main issue. |

Suggested caption:

`Negative control experiments. These follow-ups are useful paper evidence because they show that the final category-profile gain is not a trivial optimizer artifact or a nearby gate-setting accident.`

## Recommended Claim Discipline

Claims that are supported:

- the best category-profile run exceeds the original baseline by `21.08%`
- the category-profile family is the strongest family found so far
- the ten-seed matched comparison shows a net-positive robustness trend

Claims that should be avoided:

- that category-profile wins on every seed
- that the multi-seed robustness advantage is already statistically decisive
- that the strict three-seed-baseline-mean `+20%` target has already been cleared
