# Roadmap / TODO

Deferred work, roughly in priority order. Each item notes *why* and *where it
plugs in*, so it can be picked up cold.

## Scorer layers (open-ended correctness)

The deterministic `structured` scorer (`src/avbench/eval/scorer.py`) is
conservative: the synthetic-control harness measures **corrupt-catch ≈ 0.56**,
i.e. it misses subtle single-word corruptions because token-F1 overlap stays
high. Two layers plug in behind the same `Scorer` interface
(`@register_scorer(...)`), nothing downstream changes:

- [ ] **Layer 2 — NLI claim-entailment scorer.** Decompose the reference into
  atomic claims; mark each entailed/contradicted by the prediction (e.g.
  DeBERTa-MNLI). Deterministic, no generative self-preference. Directly targets
  the corrupt-catch weakness. Validate with `scripts/validate_scorer.py`.
- [ ] **Layer 3 — confidence-blind, cross-family LLM judge.** Reference-grounded,
  model-anonymized, rubric + CoT, binary verdict; use a *different* model family
  than the candidate VLM. For the genuinely open-ended residual only.
- [ ] **Human validation set.** When available, hand-label ~100–200 `(ŷ, y)` pairs
  to report scorer–human agreement (Cohen's κ) and tune thresholds. The synthetic
  harness is the stopgap until then.

## Metrics

- [ ] **Selective prediction / AURC** — risk–coverage curve, area under it,
  accuracy@coverage. Ties to the new `abstention` / `self_reflection` strategies
  and the remote-assistance thread.
- [ ] **VAUQ** — "how much of the confidence comes from vision" (proposal O2-KR2).
  Related to the `vl_uncertainty` strategy's visual perturbation.
- [ ] **Brier score + reliability diagram** — standard calibration companions;
  the binned data already exists in `eval/metrics.py`.

## Datasets

- [ ] **nuScenes-QA / DriveLM adapters** — reuse `fetch_images.py` (same nuScenes
  images) and the adapter pattern. nuScenes-QA is template-based (short answers →
  cleaner exact-match correctness).

## Strategies

- [ ] **vl_uncertainty: prompt-perturbation half** — current impl perturbs images
  only; the paper also perturbs the prompt. Also upgrade answer clustering from
  text-match to entailment-based once Layer 2 exists.

## Runs

- [ ] **Real-Gemini strategy comparison** once daily free quota resets — compare
  direct / verbal_confidence / consistency / self_reflection / abstention /
  vl_uncertainty on the behavior-MCQ split (clean exact-match correctness) to see
  which confidence signal calibrates best (lowest ECE / highest AUROC).
