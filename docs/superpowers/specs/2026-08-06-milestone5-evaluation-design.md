# Milestone 5: Evaluation, Modality Ablations, Calibration, Abstention

Date: 2026-08-06
Status: Approved (autonomous run — same process note as M3/M4's specs)

## Context

`CLAUDE.md`'s M5 goal: "evaluation, modality ablations, calibration, abstention." Unlike M2-M4 (each centered on one new fusion architecture), M5 is about evaluation *methodology* — and it directly implements a founding principle stated at the top of this project's own `CLAUDE.md` that no milestone has touched yet: *"Keep outputs honest about uncertainty; the model should be able to abstain."*

M5 also gets to apply, deliberately and systematically, the lessons M2/M3/M4's final reviews each had to catch by hand:
- M2: a headline claim ("scaling hurts") was disproven by an ablation the notebook hadn't run.
- M3: a headline claim ("shared/private decomposition drives the improvement") was disproven by a controlled ablation + regularization sweep; single-seed results were later shown to be conservative, not inflated, once multi-seed checked.
- M4: a headline premise ("contiguous dialogue") was factually wrong; a "no context" ablation the milestone hadn't originally run showed dialogue context contributes nothing measurable; an "identical capacity" claim was false; a cross-milestone comparison was baseline-blind.

M5 builds those habits into the process up front instead of relying on a reviewer to catch their absence after the fact: multi-seed or bootstrap uncertainty by default, an explicit trivial baseline on every reported number, and no single-run headline claims.

## Data scope: reuse M4's dialogue dataset, target-only features

M4's `data/dialogue/` (512 real clips, 10 episodes, already downloaded and embedded) is reused as-is — no new acquisition. M5 uses only each row's **target** utterance features (not its dialogue context): M4's own final review already established that context doesn't measurably help on this data, so including it here would conflate "does modality X help" with "does context help," muddying exactly the kind of question M5 exists to answer cleanly. Audio `(8,768)` and video `(5,512)` sequence embeddings are mean-pooled down to single `(768,)`/`(512,)` vectors per utterance — the same pooling M2 used — since the modality-ablation classifiers here are simple, not sequence models.

**Same episode-level split as M4** (`TRAIN_EPISODES`/`VAL_EPISODES`/`TEST_EPISODES`, 7/1/2 episodes, 365/40/107 rows) — for direct comparability against M4's own numbers on the identical held-out test set, even though a plain row-level split would also be leak-free once dialogue context is dropped.

## Modality ablations: 7-way sweep, bootstrap confidence intervals (not seeds)

All 7 non-empty subsets of `{text, audio, video}`: T, A, V, TA, TV, AV, TAV. Each is `np.hstack` concatenation of the pooled features for whichever modalities are included, `StandardScaler`, `LogisticRegression` — the same simple classifier family M2 used, deliberately: this milestone's question is "which modalities carry signal," not "does a fancier architecture help" (M2-M4 already explored that axis).

**Uncertainty via bootstrap resampling of the fixed 107-row test set (1,000 resamples), not multiple training seeds.** `LogisticRegression` with a fixed solver on a fixed training set is deterministic (no random initialization to average over, unlike M3/M4's neural networks) — refitting it 5 times would produce 5 identical numbers, which is not useful uncertainty quantification and would be performing the *appearance* of the multi-seed lesson without its substance. Bootstrapping the test set instead gives a genuine confidence interval on the reported accuracy/macro-F1 for a fixed, already-fitted classifier — the statistically appropriate technique for this specific situation, not a mechanical reapplication of "always use 3-5 seeds" where it doesn't fit.

Each combination's result is reported as point estimate + 95% bootstrap CI, alongside the majority-class baseline (recomputed from the actual test split, not assumed) — extending M4's fix-wave lesson (always report a baseline) to every number, not just the headline one.

## Calibration

Take the best-performing combination from the ablation sweep (whichever it turns out to be — not decided in advance) as the "final model" for calibration analysis. Using its `predict_proba` output on the test set:
- **Reliability diagram**: bin predictions by predicted-class confidence (e.g. 10 bins), plot mean confidence vs. actual accuracy within each bin — a perfectly calibrated model has points on the diagonal.
- **Expected Calibration Error (ECE)**: the bin-size-weighted average gap between confidence and accuracy across bins.

**Explicit small-sample caveat, stated up front, not discovered by a reviewer:** 107 test rows split across up to 10 confidence bins means some bins will have only a handful of points — ECE and the reliability diagram here are a first, coarse pass, not a statistically powerful calibration study. This is said plainly in the notebook rather than presented as more precise than it is.

## Abstention

Same final model, same test set. For a sweep of confidence thresholds (e.g. 0.0 to 1.0 in steps), compute:
- **Coverage**: fraction of test samples whose top predicted-class probability exceeds the threshold (i.e., the model would answer rather than abstain).
- **Risk (error rate) among answered samples**: accuracy restricted to the covered subset.

Report as a risk-coverage table/plot — this is the direct, concrete implementation of `CLAUDE.md`'s "the model should be able to abstain" principle: showing that restricting to higher-confidence predictions measurably improves accuracy among the answered subset (or honestly reporting if it doesn't, on this data).

## Cross-milestone evaluation summary

A consolidated, honest comparison table across every milestone's real results (M0 toy, M1 real text, M2 concatenation, M3 MISA, M4 cross-modal, M5's own best ablation combination), each with its **own** trivial baseline and each number's accuracy expressed as a multiple of that baseline (directly extending the "M4 vs M3 baseline-blind comparison" lesson to the full project history, not just the newest pair). This is presented once, clearly labeled as directional (different milestones used different data subsets/scales), not as a leaderboard implying a controlled comparison that was never run.

## New files

- `src/modality_ablation.py` — loads M4's dialogue data, builds target-only pooled features, runs the 7-way sweep with bootstrap CIs, saves results.
- `src/calibration_abstention.py` — takes the ablation's best combination, computes the reliability diagram data / ECE, and the risk-coverage sweep.
- `notebooks/milestone5_evaluation.ipynb` — narrated version of both scripts plus the cross-milestone summary table, explaining calibration/abstention/bootstrap-CI/ECE in plain language for a beginner (all new terms for this project).

## Error handling

Same pattern as every prior milestone: `SystemExit` with a clear pointer if `data/dialogue/index.csv` or its embeddings are missing.

## Testing / validation

No pytest suite (project convention).
- `modality_ablation.py`: run it, confirm 7 rows printed with point estimates + CIs, confirm the baseline is recomputed from real data (not hardcoded), report honestly whichever combination wins — don't assume `TAV` wins just because it uses the most information.
- `calibration_abstention.py`: run it, confirm ECE is a finite number in `[0, 1]`, confirm the risk-coverage table shows accuracy generally non-decreasing as coverage decreases (a basic sanity check — if higher-confidence subsets are somehow *less* accurate, that's a real, reportable, surprising finding, not a bug to hide).
- Notebook: execute top-to-bottom, zero errors.

## Explicitly out of scope

- New data acquisition (full 2,213-clip MIntRec set, or anything beyond M4's existing 512 clips).
- Calibration techniques beyond a reliability diagram + ECE (e.g. temperature scaling / Platt scaling recalibration) — measuring calibration, not fixing it, is this milestone's scope; recalibration is a natural M6+ extension if pursued.
- Any change to M0-M4's existing files.
