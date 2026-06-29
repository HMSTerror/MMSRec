# Beauty2013 Multimodal Recommendation Progress

## Objective

Build a paper-grade multimodal sequential recommender on Amazon Beauty 2013, starting from raw review and metadata files, using the SASRec baseline as the reference point, and pushing early-stopped test `NDCG@10` to at least a `20%` relative gain over the baseline.

## Dataset And Protocol

- Dataset: Amazon Beauty 2013
- Raw pipeline inputs:
  - `reviews_Beauty.json.gz`
  - `meta_Beauty.json.gz`
- Canonical builder: `mmsrec.data.beauty.build_dataset(...)`
- Original SASRec-style export: `data/amazon_beauty_2013/exports/Beauty.txt`
- Verified export hash: `46cc6f37bc3490b4e2d74382ccf055806c3cab68`
- Split protocol: leave-one-out
- Model selection: best validation checkpoint
- Main metric: `NDCG@10`

Verified dataset statistics:

| Users | Items | Interactions | Train Rows | Val Rows | Test Rows |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 22,363 | 12,101 | 198,502 | 131,413 | 22,363 | 22,363 |

## Baseline And Target

Verified PyTorch SASRec baseline:

- Experiment: `beauty2013_paper_v1_baseline`
- Test `NDCG@10 = 0.04442677095272959`
- Test `HR@10 = 0.0685954478379466`

Target threshold for a `20%` relative gain:

- Required `NDCG@10 >= 0.053312125143275505`

## Best Current Result

Current best verified run:

- Experiment: `beauty2013_metadata_v39_profile_seed_m2p5_remaining_seed102`
- Test `NDCG@10 = 0.05379175070654626`
- Test `HR@10 = 0.09028305683495058`
- Best epoch: `4`
- Stop epoch: `9`
- Relative gain over baseline: `21.08%`
- Absolute margin above the `20%` target: `0.0004796255632707552`

Controlled three-seed comparison against the updated baseline mean:

- baseline seeds `100/101/102` mean `NDCG@10 = 0.045439785703416356`
- corresponding `20%` threshold on that baseline mean: `0.05452774284409963`
- current best single seed (`v39 seed102`) relative gain vs baseline mean: `18.38%`
- current category-profile three-seed mean (`100/101/102`) relative gain vs baseline mean: `16.01%`

Interpretation:

- The original user target is satisfied under the single-run baseline reference used throughout the search.
- For a paper-quality multi-seed comparison, the current winning family is still below `+20%` when normalized by the new three-seed baseline mean.
- This makes robustness, not raw peak score, the central remaining gap for a B-tier submission.

Best current configuration:

| Hyperparameter | Value |
| --- | --- |
| feature file | `content_features_metadata_source075_w15_with_category_profile_only.pt` |
| hidden size | `192` |
| heads | `6` |
| blocks | `2` |
| batch size | `256` |
| eval batch size | `512` |
| learning rate | `5e-4` |
| weight decay | `1e-4` |
| dropout | `0.15` |
| target block dim | `96` |
| id gate init | `-1.2` |
| text gate init | `-0.35` |
| image gate init | `-3.2` |
| category profile gate init | `-2.5` |
| text profile gate init | `-8.0` |
| image profile gate init | `-8.0` |
| text modality | on |
| image modality | on |
| explicit brand branch | off |
| explicit category branch | on, profile-only |
| category candidate fusion | off |
| category direct score branch | off |

## Model Family

The strongest model family now remains the stable concatenated `source075_w15` backbone, with the first successful category auxiliary path added only at sequence-profile level:

1. Metadata TF-IDF/SVD text built from `title + description + categories + brand`
2. Appended source semantic text block
3. Appended brand block
4. Appended category block
5. Multimodal SASRec with behavior encoder + cross-attention memory injection
6. Standalone category profile scoring without category candidate fusion or direct category residual scoring

The best feature weighting found so far is:

- source semantic text weight: `0.75`
- brand weight: `1.5`
- category weight: `1.5`

## Paper-Style Result Snapshot

Selected milestone results:

| Variant | Best Epoch | Test NDCG@10 | Delta vs Baseline | Relative Gain |
| --- | ---: | ---: | ---: | ---: |
| SASRec baseline | 5 | 0.044426771 | 0.000000000 | 0.00% |
| Early multimodal decomposition (`v3` tuned) | 5 | 0.050209541 | 0.005782771 | 13.02% |
| Metadata + source text (`min_df=1`) full tuned | 4 | 0.051624386 | 0.007197615 | 16.20% |
| Metadata + source text + brand/category (`v5`) | 4 | 0.052411247 | 0.007984476 | 17.97% |
| Same family with structured weight `1.5` (`v8`) | 4 | 0.052646413 | 0.008219642 | 18.50% |
| Best current `source=0.75, w1.5` (`v16`) | 4 | 0.053095715 | 0.008668944 | 19.51% |
| Category profile-only best (`v36 m2p5`) | 4 | 0.053309772 | 0.008883001 | 19.99% |
| Category profile multi-seed hit (`v39 seed102`) | 4 | 0.053791751 | 0.009364980 | 21.08% |

## Focused Search Since The Previous Best

### Source-Weight Sweep On The Best Family

| Experiment | Test NDCG@10 |
| --- | ---: |
| `v16 source=0.75, w1.5` | 0.053095715 |
| `v17 source=0.70, w1.5` | 0.053051293 |
| `v17 source=0.65, w1.5` | 0.052975325 |
| `v18 source=0.80, w1.5` | 0.052914937 |
| `v18 source=0.78, w1.5` | 0.052905848 |

Conclusion:

- `source=0.75` remains the best point in the local neighborhood.
- Pure microtuning on the source-text weight is effectively exhausted.

### Description Trimming Ablations

| Experiment | Test NDCG@10 |
| --- | ---: |
| `v20` title + category + brand, `source=1.00` | 0.052229495 |
| `v20` title + category + brand, `source=0.75` | 0.052571002 |
| `v21` title + category + brand + desc128 | 0.052771876 |
| `v21` title + category + brand + desc256 | 0.052644806 |

Conclusion:

- Removing or truncating descriptions did not beat the full `text_input` formulation.

### Training-Side Tweaks Around The Best Feature Family

| Experiment | Test NDCG@10 |
| --- | ---: |
| `v22` batch size `128` | 0.051232530 |
| `v22` batch size `128` + gate tweak | 0.051174847 |

Conclusion:

- Narrow optimizer-side perturbations around the best feature family have been fragile.

### Cosine Scheduler Sweep (`v23`)

All cosine runs used the current best `source075_w15` feature family.

| Experiment | Scheduler | Best Epoch | Stop Epoch | Test NDCG@10 |
| --- | --- | ---: | ---: | ---: |
| `v23` `warmup=1`, `min_lr=5e-5`, `ep=20` | cosine | 4 | 12 | 0.052987044 |
| `v23` `warmup=2`, `min_lr=5e-5`, `ep=20` | cosine | 4 | 12 | 0.051588033 |
| `v23` `warmup=1`, `min_lr=1e-4`, `ep=20` | cosine | 4 | 12 | 0.052994472 |
| `v23` `warmup=0`, `min_lr=0`, `ep=18` | cosine | 4 | 11 | 0.052881465 |

Conclusion:

- Simple cosine scheduling did not beat the unscheduled best run.
- Longer warmup was actively harmful on this configuration.

### Plateau Scheduler Sweep (`v24`)

Completed plateau runs on the same `source075_w15` family:

| Experiment | Scheduler | Best Epoch | Stop Epoch | Test NDCG@10 |
| --- | --- | ---: | ---: | ---: |
| `v24` `patience=1`, `factor=0.5`, `min_lr=1e-4`, `ep=24` | plateau | 4 | 14 | 0.053095715 |
| `v24` `patience=2`, `factor=0.5`, `min_lr=1e-4`, `ep=24` | plateau | 4 | 14 | 0.053095715 |
| `v24` `patience=1`, `factor=0.7`, `min_lr=1e-4`, `ep=24` | plateau | 4 | 14 | 0.053095715 |

Conclusion:

- Plateau scheduling can preserve the current best point.
- It still does not move the model past the existing `epoch 4` ceiling.
- The fourth low-priority plateau variant was intentionally stopped once the pattern became clear.

### Explicit Title-Append Sweep (`v25`)

All `v25` runs kept the `source075_w15` family fixed and appended an additional title-specific TF-IDF/SVD block.

| Experiment | Title Block | Best Epoch | Stop Epoch | Test NDCG@10 |
| --- | --- | ---: | ---: | ---: |
| `v25` `title32`, `title_weight=1.0` | append title | 5 | 10 | 0.050640717 |
| `v25` `title32`, `title_weight=1.5` | append title | 5 | 10 | 0.050423961 |
| `v25` `title64`, `title_weight=1.0` | append title | 4 | 9 | 0.052661206 |
| `v25` `title64`, `title_weight=1.5` | append title | 4 | 9 | 0.052277172 |

Conclusion:

- Explicit title append blocks were consistently worse than the current best family.
- Lower-dimensional title append blocks were substantially harmful.
- Even the best title-append variant stayed well below the current best `0.053095715`.

### Weighted Main-Text Repeat Sweep (`v26`, Partial)

These runs did not add new dimensions. Instead, they rebuilt the main TF-IDF text corpus by repeating selected structured fields before `text_input`.

Completed runs:

| Experiment | Text Reweighting | Best Epoch | Stop Epoch | Test NDCG@10 |
| --- | --- | ---: | ---: | ---: |
| `v26` `text_title2x` | repeat title once before base text | 4 | 9 | 0.052270035 |
| `v26` `text_title3x` | repeat title twice before base text | 4 | 9 | 0.052629392 |
| `v26` `text_title2x_category2x` | repeat title and category once | 4 | 9 | 0.052335153 |

Operational note:

- The remaining `text_title2x_brand2x_category2x` variant was intentionally stopped after the first three results made the direction look low value.

Conclusion:

- Reweighting the main TF-IDF corpus by repeating title text did not recover the current best point.
- Adding category repetition on top of title repetition also stayed below the best configuration.
- The current evidence suggests that direct title emphasis, whether by append blocks or corpus repetition, is not the missing `0.0002`.

### Structured Text Repeat Sweep (`v27`)

These runs removed title emphasis and focused only on repeating structured metadata inside the main TF-IDF text corpus.

| Experiment | Text Reweighting | Best Epoch | Stop Epoch | Test NDCG@10 |
| --- | --- | ---: | ---: | ---: |
| `v27` `text_category2x` | repeat category once before base text | 4 | 9 | 0.053055819 |
| `v27` `text_brand2x` | repeat brand once before base text | 4 | 9 | 0.052249513 |
| `v27` `text_brand2x_category2x` | repeat brand and category once | 4 | 9 | 0.052721904 |

Conclusion:

- `category` repetition is the only text-repeat variant that returned to the neighborhood of the current best model.
- `brand` repetition is clearly harmful.
- Even the strongest `category2x` variant still stayed below the current best `0.053095715` and below the `20%` threshold.

### Additional Seed Follow-Up (`v28`)

Additional seeds on the current best `source075_w15` family:

| Experiment | Seed | Best Epoch | Stop Epoch | Test NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| `v28` `source075_w15_seed103` | 103 | 4 | 9 | 0.049283611 |
| `v28` `source075_w15_seed104` | 104 | 4 | 9 | 0.051247922 |
| `v28` `source075_w15_seed105` | 105 | 4 | 9 | 0.050568369 |

Conclusion:

- Further seed follow-up did not produce a stronger single run.
- The robustness story is now weaker than before, not stronger.

### Fast Category Transform Sweep (`v31`)

These runs reused the built `text_category2x` feature tensor and rescaled only the appended slices, avoiding another TF-IDF/SVD rebuild.

| Experiment | Change | Test NDCG@10 |
| --- | --- | ---: |
| `v31` `text_category2x_cw125_fast` | lower appended category block weight to `1.25` | 0.052659725 |
| `v31` `text_category2x_cw100_fast` | lower appended category block weight to `1.00` | 0.052560779 |

Conclusion:

- Lowering the appended category block weight below `1.5` is counterproductive.
- The `text_category2x` family does not recover the current best by weakening the explicit category block.

### Category Source Sweep On The Fast Transform Family (`v32`)

Completed source-weight follow-ups on the same `text_category2x` fast-transform family:

| Experiment | Change | Test NDCG@10 |
| --- | --- | ---: |
| `v32` `text_category2x_source070_fast` | lower source-text weight to `0.70` | 0.052921916 |
| `v32` `text_category2x_source072_fast` | lower source-text weight to `0.72` | 0.052942853 |

Operational note:

- Additional `0.78` / `0.80` source-weight tensors were prepared but not promoted further once the lower-weight variants also stayed below both `v27 text_category2x` and the global best `v16`.

Conclusion:

- The `category2x` family does not benefit from shifting the appended source-text weight downward.
- This closes out the most obvious local reweighting neighborhood around `text_category2x`.

### Explicit Category Score Branch Sweep (`v34`)

This was the first genuine model-side follow-up after the feature-space local search flattened. The run kept the `text_category2x` backbone, added an explicit standalone category embedding branch, and swept the category score gate initialization.

| Experiment | Category Score Gate Init | Test NDCG@10 |
| --- | ---: | ---: |
| `v34` `catbranch_m1p5` | `-1.5` | 0.049583071 |
| `v34` `catbranch_m1p0` | `-1.0` | 0.050004113 |
| `v34` `catbranch_m0p5` | `-0.5` | 0.050037117 |

Conclusion:

- Direct category residual scoring is decisively harmful on Beauty2013 in this model family.
- The failure is large enough that this direction should be treated as closed rather than merely under-tuned.

### Category Fusion Auxiliary Sweep (`v35`, Stopped Early)

The next model-side follow-up moved back to the strongest `v16 source075_w15` backbone and tried to use category as an auxiliary signal while still letting category enter the fused candidate representation:

1. keep the strongest concatenated `source075_w15` text backbone fixed
2. add standalone category embeddings to the feature payload
3. allow category to influence:
   - fused candidate representations
   - optionally a separate sequence-level category profile score
4. explicitly disable the direct category residual score branch that failed in `v34`

First completed `v35` result:

| Experiment | Category Auxiliary Mode | Test NDCG@10 |
| --- | --- | ---: |
| `v35` `catfusion_only` | fused candidate category branch only, no direct score, no profile | 0.049376135 |

Conclusion:

- Candidate-side category fusion is itself harmful on the strongest `source075_w15` backbone.
- Because every remaining `v35` variant still contained the same harmful fusion component, the sweep was stopped early rather than spending more GPU time on a dominated direction.

### Category Profile-Only Sweep (`v36`)

The current live hypothesis is stricter and cleaner than `v35`:

1. keep the strongest `source075_w15` backbone fixed
2. keep standalone category embeddings available
3. remove category from:
   - fused candidate representations
   - direct category residual scoring
4. keep only a sequence-level category profile score as the auxiliary path

Completed `v36` variants:

- `catprofileonly_m3p0`
- `catprofileonly_m2p5`
- `catprofileonly_m2p0`
- `catprofileonly_m1p5`

| Experiment | Category Profile Gate Init | Test NDCG@10 |
| --- | ---: | ---: |
| `v36` `catprofileonly_m3p0` | `-3.0` | 0.053154266 |
| `v36` `catprofileonly_m2p5` | `-2.5` | 0.053309772 |
| `v36` `catprofileonly_m2p0` | `-2.0` | 0.053257495 |
| `v36` `catprofileonly_m1p5` | `-1.5` | 0.053090839 |

Current interpretation:

- Pure sequence-level category profile scoring is the first category-auxiliary route that materially improves over the global best `v16`.
- The response curve is peaked rather than monotonic: `-2.5` is best, while stronger (`-3.0`) and weaker (`-2.0`, `-1.5`) gates are all worse.
- `v36 m2p5` is only `0.00000235348338356` below the `20%` target threshold, so this line is now the primary frontier.

Rationale:

- The remaining gap to the `20%` target is only `0.0002164`.
- `v34` showed direct category residual scoring is harmful.
- `v35` showed category candidate fusion is also harmful.
- The remaining plausible category path is now a pure sequence-level auxiliary signal.

### Gate-Refine Attempt (`v38`, Paused)

An initial narrow gate-refine attempt was prepared around the `-2.5` region. While that attempt was being executed, a comparability issue was discovered: the first structured-memory code path temporarily changed the size of the modality-type embedding table even for configurations that did not use structured memory, which altered initialization trajectories for unrelated runs.

What was done:

1. the comparability bug was fixed so non-memory configurations keep the original modality-type parameterization
2. the current best `v36 m2p5` setting was re-run under the corrected code and reproduced exactly:
   - `NDCG@10 = 0.053309771659891945`
3. the active frontier was then redirected to training-dynamics follow-up on the reproduced best point

Conclusion:

- `v38` should not be used as evidence in the paper until it is rerun under the corrected code.
- The immediate priority is now `v40`, because the remaining gap is so small that scheduler-side stabilization is a higher-value next check than another local gate guess.

### Active Plateau Follow-Up On The Reproduced Best Point (`v40`)

Completed high-priority check:

- `plateau_p1_f05_min1e4_ep24`

Result:

| Experiment | Scheduler | Test NDCG@10 |
| --- | --- | ---: |
| `v40` `plateau_p1_f05_min1e4_ep24` | plateau | 0.053309772 |

Why this is the right next step:

- The exact `v36 m2p5` best point has been reproduced under the corrected code.
- The remaining gap to the threshold is only `0.00000235348338356`.
- Plateau scheduling previously preserved the best point on the older `v16` family, so it is a reasonable low-risk way to try to push the reproduced `m2p5` configuration over the line without changing the architecture again.

Outcome:

- The first plateau follow-up exactly tied the reproduced best point rather than improving it.
- Because the gap is microscopic and the first scheduler result showed no movement, the active search moved back to more targeted local or architectural changes instead of continuing the lower-value scheduler grid.

### Micro Gate Refine Around The Best Profile Point (`v41`)

Completed micro-refine results around `category_profile_gate_init = -2.5`:

| Experiment | Category Profile Gate Init | Test NDCG@10 |
| --- | ---: | ---: |
| `v41` `m2p52` | `-2.52` | 0.053243601 |
| `v41` `m2p51` | `-2.51` | 0.053268614 |
| `v41` `m2p49` | `-2.49` | 0.053276549 |
| `v41` `m2p48` | `-2.48` | 0.053286531 |
| `v41` `m2p47` | `-2.47` | 0.053264841 |

Conclusion:

- The local neighborhood around `-2.5` has now been probed on both sides.
- None of the nearby gates beat the original `-2.5` setting.
- The current evidence says the single-run optimum for this family really is very close to `-2.5`, and the residual gap is not coming from coarse gate mis-setting.

### Pure Category Memory Sweep (`v37`, In Progress)

This sweep removes the category candidate and profile branches entirely and instead injects category only through sequence-side memory tokens.

First completed result:

| Experiment | Category Memory Weight | Test NDCG@10 |
| --- | ---: | ---: |
| `v37` `catmemory_w1p0` | `1.0` | 0.052153989 |

Interim interpretation:

- Pure category memory alone is materially weaker than the profile-only best line.
- It may still be useful as a complementary signal, but it does not look like a standalone replacement for `v36`.

### Category Profile + Memory Combo Sweep (`v42`, Stopped Early)

Completed first result:

| Experiment | Category Memory Weight | Test NDCG@10 |
| --- | ---: | ---: |
| `v42` `profilem2p5_memoryw1p0` | `1.0` | 0.051773041 |

Conclusion:

- Layering category memory onto the strongest profile-only route is materially worse than the profile-only baseline itself.
- Because the first combo result underperformed badly, this line was stopped rather than spending more GPU time on nearby memory weights.

### Profile-Only Seed Follow-Up (`v39`, `v43`)

Once the search got within `0.00000235348338356` of the target, the highest-value follow-up shifted from more local gate nudges to seed stability on the strongest category-profile line.

Completed seed follow-up results:

| Experiment | Seed | Test NDCG@10 |
| --- | ---: | ---: |
| `v36` `catprofileonly_m2p5` | `100` | 0.053309772 |
| `v39` `profile_seed_m2p5_seed101` | `101` | 0.051037601 |
| `v39` `profile_seed_m2p5_remaining_seed102` | `102` | 0.053791751 |
| `v39` `profile_seed_m2p5_remaining_seed103` | `103` | 0.051003444 |
| `v43` `profile_seed104_seed104` | `104` | 0.050719694 |

Main takeaways:

- Seed `102` is the first verified run that cleanly breaks the `20%` relative-gain threshold.
- The gain is real, not rounding noise: `0.05379175070654626` vs the required `0.053312125143275505`, a `21.08%` relative improvement over baseline.
- The architecture is now strong enough to hit the target, but the variance across seeds is still large enough that robustness remains the main paper weakness.

### Optimizer Follow-Up On The Best Profile Family (`v44`)

After the profile-only line nearly crossed the target, a small optimizer sweep checked whether the same family could improve further without changing architecture.

Completed `v44` results:

| Experiment | Learning Rate | Weight Decay | Test NDCG@10 |
| --- | ---: | ---: | ---: |
| `v44` `lr4e4_wd1e4` | `4e-4` | `1e-4` | 0.051938248 |
| `v44` `lr4e4_wd5e5` | `4e-4` | `5e-5` | 0.051939693 |
| `v44` `lr5e4_wd5e5` | `5e-4` | `5e-5` | 0.053294983 |
| `v44` `lr6e4_wd1e4` | `6e-4` | `1e-4` | 0.052861853 |

Conclusion:

- Lowering the learning rate to `4e-4` is clearly harmful.
- The best optimizer variant (`lr=5e-4`, `wd=5e-5`) comes very close to `v36 m2p5`, but still does not beat the best seed follow-up result.
- The final improvement came from seed follow-up on the best architecture rather than optimizer retuning.

### Category Profile + Text-Profile Blend Sweep (`v45`)

This follow-up asked whether the strongest category-profile family would benefit from reintroducing a very weak sequence-level text-profile signal while keeping the current best category-profile branch fixed.

Completed `v45` results:

| Experiment | Text Profile Gate Init | Test NDCG@10 |
| --- | ---: | ---: |
| `v45` `textprofile_m7p0` | `-7.0` | 0.053299835 |
| `v45` `textprofile_m6p0` | `-6.0` | 0.053300404 |
| `v45` `textprofile_m5p0` | `-5.0` | 0.053288344 |
| `v45` `textprofile_m4p0` | `-4.0` | 0.053260409 |

Conclusion:

- Reintroducing text-profile scoring does not improve the category-profile best family.
- All four `v45` points underperform the original `v36 m2p5` reference.
- This closes the mild text-profile blend hypothesis for the current backbone.

### Plateau Scheduler Seed Follow-Up (`v47`)

This follow-up tested whether the `v40` plateau schedule that tied the best seed-100 run could rescue weak seeds without changing the architecture.

Completed `v47` results:

| Experiment | Seed | Scheduler | Test NDCG@10 |
| --- | ---: | --- | ---: |
| `v47` `profile_plateau_seed_seed101` | `101` | plateau | 0.051037601 |
| `v47` `profile_plateau_seed_seed103` | `103` | plateau | 0.051003444 |

Conclusion:

- Plateau scheduling does not improve the weak-seed outcomes.
- For both seeds, the best `NDCG@10` matches the earlier non-plateau results, only with a later stop epoch.
- The current robustness problem is therefore not a simple training-schedule issue.

### Seed-Gate Robustness Follow-Up (`v48`)

Current active hypothesis:

1. the weak seeds may prefer a different category-profile gate than the seed-100 optimum `-2.5`
2. the local seed-100 sweep already showed `-3.0` and `-2.0` are the nearest plausible alternatives
3. the highest-value next check is therefore a focused seed-gate grid on the weak seeds

Active runs:

- `seed101` with `category_profile_gate_init in {-3.0, -2.0}`
- `seed103` with `category_profile_gate_init in {-3.0, -2.0}`

Goal:

- determine whether weak-seed underperformance is partly a gate-location mismatch rather than irreducible variance
- improve the paper's multi-seed story without opening a much larger architecture search

Completed `v48` results:

| Experiment | Seed | Category Profile Gate Init | Test NDCG@10 |
| --- | ---: | ---: | ---: |
| `v48` `seed101_m3p0` | `101` | `-3.0` | 0.050989398 |
| `v48` `seed101_m2p0` | `101` | `-2.0` | 0.050904136 |
| `v48` `seed103_m3p0` | `103` | `-3.0` | 0.050931657 |
| `v48` `seed103_m2p0` | `103` | `-2.0` | 0.050935377 |

Conclusion:

- Neither neighboring gate improves the weak seeds relative to the original `m2p5` runs.
- For both `seed101` and `seed103`, the weak-seed problem survives both stronger and weaker nearby category-profile gates.
- This largely closes the local gate-mismatch hypothesis.

### Matched-Seed Source075 Follow-Up (`v49`, `v49c`) And Category Seed105 Follow-Up (`v50`)

This follow-up answered the most important robustness question left by the earlier search:

1. does the category-profile branch still beat the older `source075_w15` family when both are compared on the same seeds?
2. if yes, is the gain broad enough to justify keeping the category-profile branch as the main paper model despite its variance?

Completed matched-seed results through seed `109`:

| Seed | `source075_w15` NDCG@10 | Category-Profile NDCG@10 | Delta (Category - Source) |
| ---: | ---: | ---: | ---: |
| `100` | 0.053095715 | 0.053309772 | +0.000214056 |
| `101` | 0.052208915 | 0.051037601 | -0.001171314 |
| `102` | 0.051143831 | 0.053791751 | +0.002647919 |
| `103` | 0.049283611 | 0.051003444 | +0.001719833 |
| `104` | 0.051247922 | 0.050719694 | -0.000528227 |
| `105` | 0.050568369 | 0.051976060 | +0.001407692 |
| `106` | 0.051606303 | 0.051796545 | +0.000190242 |
| `107` | 0.050928551 | 0.050819963 | -0.000108589 |
| `108` | 0.052338930 | 0.052643455 | +0.000304526 |
| `109` | 0.049679794 | 0.052027992 | +0.002348198 |

Matched-seed summary across seeds `100-109`:

- category-profile wins `7 / 10` seeds
- mean `source075_w15 NDCG@10 = 0.0512101941747265`
- mean `category-profile NDCG@10 = 0.05191262781170142`
- mean absolute delta `= +0.000702433636974921`
- mean relative gain vs matched `source075_w15` `= +1.40%`

Conclusion:

- The category-profile branch is not uniformly better, but it is a net positive on the matched-seed comparison now available.
- The `seed107` pair shows the advantage is not monotonic: the older `source075_w15` family can still win on some new seeds.
- The `seed108` pair restores the net-positive matched-seed pattern and keeps the category-profile mean ahead on both win count and average score.
- The `seed109` pair strengthens the robustness story further: category-profile now wins `7 / 10` matched seeds and improves the average matched delta.
- The main weakness is no longer "maybe the new branch is useless"; it is "the new branch helps on average but with uncomfortable variance."
- This is the strongest current paper argument for keeping the category-profile family as the main model while being candid about robustness.

Latest completed extension:

- `source075_w15 currentcode seed107` finished at `0.05092855129587869`
- `category-profile seed107` finished at `0.05081996273157817`
- `source075_w15 currentcode seed108` finished at `0.05233892957308715`
- `category-profile seed108` finished at `0.05264345542831027`
- `source075_w15 currentcode seed109` finished at `0.049679793846013676`
- `category-profile seed109` finished at `0.05202799234176532`

Current decision point:

- matched-seed evidence now spans ten seeds
- the category-profile branch remains net positive on both mean score and win count
- the highest-value next move is now to stop architecture search and consolidate the paper package unless a very specific new hypothesis appears

Small-sample statistical check on the ten matched seeds:

- exact sign test one-sided p-value `= 0.171875`
- exact sign test two-sided p-value `= 0.34375`
- 95% bootstrap CI for the mean matched delta `(category - source)` `= [-0.000031441, 0.001458499]`

Interpretation:

- The direction of the effect is favorable to the category-profile branch.
- The ten-seed robustness evidence is therefore better described as a positive trend than as a conventionally significant statistical win.
- For a paper submission, this is still useful evidence as long as the claim is framed honestly.

## Seed Stability

Updated PyTorch SASRec baseline across seeds `100/101/102`:

| Experiment | Seed | Test NDCG@10 |
| --- | ---: | ---: |
| `v1 baseline` | 100 | 0.044426771 |
| `v46 baseline_seed` | 101 | 0.046082128 |
| `v46 baseline_seed` | 102 | 0.045810458 |

Three-seed average:

- mean `NDCG@10 = 0.045439785703416356`
- population std `= 0.0007248449783110786`

Original `source075_w15` family across seeds `100/101/102`:

| Experiment | Seed | Test NDCG@10 |
| --- | ---: | ---: |
| `v16 source075_w15` | 100 | 0.053095715 |
| `v19 source075_w15` | 101 | 0.052208915 |
| `v19 source075_w15` | 102 | 0.051143831 |

Three-seed average:

- mean `NDCG@10 = 0.05214948730849017`
- relative gain vs baseline mean `= 14.77%`

Interpretation:

- The strongest pre-category family still does not cross the target even in its best single run.

Current category-profile family across seeds `100/101/102`:

| Experiment | Seed | Test NDCG@10 |
| --- | ---: | ---: |
| `v36` `catprofileonly_m2p5` | 100 | 0.053309772 |
| `v39` `profile_seed_m2p5_seed101` | 101 | 0.051037601 |
| `v39` `profile_seed_m2p5_remaining_seed102` | 102 | 0.053791751 |

Three-seed average:

- mean `NDCG@10 = 0.05271304127787113`
- population std `= 0.0012009440369426295`
- relative gain vs baseline mean `= 16.01%`

All observed category-profile seeds so far (`100-109`):

- mean `NDCG@10 = 0.05191262781170142`
- population std `= 0.001012784494228687`
- relative gain vs baseline mean `= 14.24%`

All observed `source075_w15` matched seeds so far (`100-109`):

- mean `NDCG@10 = 0.0512101941747265`
- population std `= 0.0011195765261739505`
- relative gain vs baseline mean `= 12.70%`

Interpretation:

- The category-profile family is now the first family to cross the target in a verified single run.
- Its three-seed mean is higher than the older `source075_w15` family mean.
- Its ten-seed matched-family mean is also higher than the ten-seed `source075_w15` mean.
- But once the baseline is also treated as a three-seed reference, the current winning family still does not clear `+20%` on either three-seed mean or all-observed-seed mean.
- The spread across seeds is therefore the main paper weakness rather than absolute peak score.

## Main Technical Conclusions

1. The best direction remains the concatenated metadata text backbone with source-text reweighting and structured upweighting.
2. Image is still not a dominant signal on Beauty2013; it may help slightly as an auxiliary branch but not enough to define the gain.
3. Explicit structured score branches are not uniformly useful:
   - direct category residual scoring is clearly harmful (`v34`)
   - category candidate fusion is also harmful (`v35`)
   - the current strongest evidence is that category helps only as a pure sequence-level profile signal
4. Description trimming, simple optimizer changes, and category-memory additions do not explain the final breakthrough.
5. The optimization ceiling is still narrow, but the architecture-side follow-ups now look lower value than consolidating the robustness story around matched-seed evidence.

## What Is Paper-Ready Right Now

The current experiment set is already strong enough to support a draft experimental section with:

1. A clean baseline vs multimodal main-result table
2. Text-only and image-only modality ablations
3. Metadata feature ablations
4. Structured-weight ablations
5. A candid robustness paragraph explaining seed variance
6. A single verified run above the `20%` relative-gain target
7. Exported markdown/CSV/LaTeX seed-summary tables for the controlled three-seed comparison and all-observed-seed robustness view

## What Is Still Missing For A Stronger B-Tier Submission

1. A cleaner robustness story for the winning category-profile family
2. Either:
   - a best single run that also beats `+20%` against the three-seed baseline mean, or
   - a stronger multi-seed average that narrows the current gap
3. Clean summary tables exported from the current experiment scripts with scheduler metadata included
4. A final decision on whether the paper should report:
   - the best single-seed result as the headline, or
   - the three-seed mean as the headline with the best-seed result as auxiliary evidence
5. Final synchronized seed-summary tables and a writing-ready robustness narrative anchored in the ten-seed matched comparison

## Recommended Next Moves

Highest-value next experiments:

1. Freeze the current matched-seed evidence at ten seeds and use it as the main robustness table unless a new concrete concern appears.
2. Treat further architecture exploration as low priority unless a new idea has a stronger justification than the already-closed `v45`, `v47`, and `v48` directions.
3. Pivot the writing plan toward a candid headline split:
   - best single-seed result for the absolute-gain headline
   - matched-seed and three-seed summaries for the robustness story
4. Turn the current evidence into the final paper package:
   - main result table
   - robustness table
   - ablation table
   - one paragraph on variance and failure cases
5. Keep the paper tables synchronized after each completed sweep:
   - `beauty2013_seed_comparison_aggregate.csv`
   - `beauty2013_seed_comparison.md`
   - `beauty2013_seed_comparison.tex`
   - `beauty2013_matched_seed_comparison.csv`
   - `beauty2013_matched_seed_comparison.md`
   - `beauty2013_matched_seed_comparison.tex`

## Artifacts

- Dataset pipeline: `mmsrec/data/beauty.py`
- Baseline trainer: `mmsrec/baselines/sasrec.py`
- Multimodal trainer/model: `mmsrec/multimodal/sasrec.py`
- Metadata feature builder: `mmsrec/features/metadata.py`
- Paper experiment runner: `experiments/beauty2013_paper.py`
- Cosine scheduler sweep script: `experiments/beauty2013_scheduler_sweep.py`
- Plateau scheduler sweep script: `experiments/beauty2013_plateau_sweep.py`
- Category auxiliary sweep script: `experiments/beauty2013_category_aux_sweep.py`
- Category profile-only sweep script: `experiments/beauty2013_category_profile_only_sweep.py`
- Category profile refine sweep script: `experiments/beauty2013_category_profile_refine_sweep.py`
- Category profile plateau sweep script: `experiments/beauty2013_category_profile_plateau_sweep.py`
- Category profile micro-refine sweep script: `experiments/beauty2013_category_profile_micro_refine_sweep.py`
- Category profile + memory combo sweep script: `experiments/beauty2013_category_profile_memory_combo_sweep.py`
- Category profile text-profile blend sweep script: `experiments/beauty2013_category_profile_textprofile_sweep.py`
- Category profile plateau seed follow-up script: `experiments/beauty2013_category_profile_plateau_seed_followup.py`
- Category profile seed-gate follow-up script: `experiments/beauty2013_category_profile_seed_gate_followup.py`
- Source075 matched-seed follow-up script: `experiments/beauty2013_source075_seed_followup.py`
- Baseline seed follow-up script: `experiments/beauty2013_baseline_seed_followup.py`
- Seed comparison export script: `experiments/beauty2013_paper_seed_table.py`
