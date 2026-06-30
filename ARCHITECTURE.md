# Architecture

Code-structure map for `avbench`. For install/usage see [README.md](README.md); for
experiment results see `local_mem/findings.md` (local-only).

_Last updated: 2026-06-30. **Update this file when modules, the data contract, scripts,
or extension points change materially.**_

## The spine: one data contract
Everything depends only on two Pydantic models in **`src/avbench/schema.py`**, so adding
a dataset, backend, or strategy never ripples downstream:

```
curate   →  data/curated/*.jsonl   Sample      (x: question+images+options, y: gold answer)
infer    →  runs/*.jsonl           Prediction  (ŷ + confidence: verbal / token_logprob)
evaluate →  join on sample_id   →   accuracy / ECE / AUROC + confusion, stratified by task
```

- **`Sample`** — `prompt_format` (`mcq` | `yesno` | `qa` | `cap`), `task_type`,
  `question`, `options`, `answer`, `images: [ImageRef]`, `object_refs`.
- **`Prediction`** — `answer`, `verbal_confidence`, `token_logprob`, `abstained`,
  `usage`, `condition` (active ablation flags), `error`.

**Separation of concerns:** curation records *data facts* (question, gold, answer-space
type, options, available images). Inference decides *how to render* them (prompt wording,
image layout, markers) and calls the model. Don't bake prompt strings or layout into the
curated `Sample` — that would break sweeping strategies/layouts over the same data.

## Layout
```
src/avbench/
  schema.py            Sample / Prediction / ImageRef / PromptFormat / TaskType — the contract
  config.py            YAML run-config; precedence CLI flag > --config > built-in default
  io_utils.py          JSONL append/read helpers

  curation/            benchmark native format → unified Sample JSONL
    base.py            DatasetAdapter ABC + registry (@register / get_adapter)
    build.py           run adapters → normalized JSONL (+ summarize)
    sensor.py          resolve QA image refs → local nuScenes paths
    stratify.py        task-balanced subsampling (per_task / total)
    adapters/
      drivebench.py    DriveBench/DriveLM: parse native JSON, lift inline MCQ choices,
                       tag task_type + prompt_format (yesno inferred from Yes/No gold)

  inference/           Sample → model call → Prediction
    client.py          VLMClient interface + backend registry (@register_backend); GenResult
    runner.py          async bounded-concurrency runner; append-JSONL; resume-on-crash
    view.py            presentation: layout (separate|stitch|single) × annotations
                       (marker_grounding); owns images_for + decorate_prompt; records condition()
    grounding.py       <c,CAM,x,y> ref parsing, marker rendering, surround stitch/order + caption
    parsing.py         extract_answer (MCQ letter / GLM box / option-text recovery),
                       confidence, abstention
    backends/          one VLMClient each — gemini (google-genai), glm + qwen
                       (OpenAI-compatible over httpx), mock (offline, deterministic)
    strategies/        confidence-elicitation methods (build_prompt + run → Prediction)
      base.py          PromptStrategy ABC + registry; render_question, answer_instruction
      direct.py        bare answer (+ token-logprob confidence when the backend exposes it)
      verbal_confidence.py   answer + self-stated 0–100
      consistency.py   k samples at T>0; answer-agreement rate = confidence
      self_reflection.py     answer, then a follow-up self-rating (also registers `abstention`)
      vl_uncertainty.py      image-perturbation sampling

  eval/                scoring + metrics (source of truth for numbers: scripts/evaluate.py)
    scorer.py          correctness scorers (exact | structured) + registry; answer_label
    metrics.py         calibration (accuracy / ECE / AUROC=conf-vs-correctness) and
                       discrimination (confusion / per-class recall / balanced accuracy)
    synthetic.py       synthetic-control harness to validate a scorer without human labels
```

## Scripts (`scripts/`, each `--config`-aware)
| script | role |
|---|---|
| `make_sample_data.py` | generate the offline synthetic fixture + images (no API key) |
| `curate.py` | adapters → `data/curated/*.jsonl` (`Sample`); `--formats mcq/yesno/qa` filter |
| `fetch_images.py` | download gated DriveBench/DriveLM nuScenes images (needs `HF_TOKEN`) |
| `infer.py` | run backend × strategy × view over a curated file → `runs/*.jsonl` (`Prediction`) |
| `preview_prompt.py` | render the exact prompt + images for one sample — no API call |
| `evaluate.py` | join curated + pred on `sample_id` → accuracy / ECE / AUROC + confusion, `--by` |
| `validate_scorer.py` | run the synthetic-control harness on a scorer |
| `_bootstrap.py` | puts `src/` on `sys.path` for the scripts |

## Extension points (add ≈ one file + a decorator)
- **New dataset** → `curation/adapters/<name>.py`, `@register("<name>")`, yield `Sample`s.
- **New model** → `inference/backends/<name>.py`, `@register_backend("<name>")`,
  implement `async generate(...) -> [GenResult]`; import it in `client.py`.
- **New confidence method** → `inference/strategies/<name>.py`, `@register("<name>")`,
  implement `build_prompt` + `run`.
- **New correctness scorer** → `@register_scorer("<name>")` in `eval/`.
- **New image presentation** → extend `view.py` (one file owns layout + prompt hook).

## Conventions
- **Metrics source of truth:** `scripts/evaluate.py`. Keep *calibration* (AUROC =
  confidence-vs-correctness) and *discrimination* (balanced accuracy / recall) separate.
- **Confidence signals by backend:** verbal (everywhere); token logprobs only Qwen
  (GLM returns `logprobs:null`; Gemini free tier none) — see `backends/glm.py` docstring.
- **Tests:** TDD; `PYTHONPATH=src python3 -m pytest -q`. Pure parsing/adapter tests run
  offline (no network).
- **Local-only / gitignored:** `data/` (except `data/sample/`), `runs/`, `experiments/`,
  `docs/`, `local_mem/`, `.env`. Experiment writeups live in
  `local_mem/experiments/` (indexed by `local_mem/findings.md`); runnable recipes sit in
  each experiment's "Repro" section.
