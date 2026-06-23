# avbench — VLM Self-Knowledge Benchmark for Autonomous Driving (prototype)

Prototype that curates a unified QA + sensor dataset from AD-VLM benchmarks, runs
a VLM (Gemini free tier to start), and records `(x, y, ŷ)` with confidence
signals — feeding a calibration analysis (ECE / AUROC, stratified by task) of
VLM self-knowledge in autonomous driving.

```
curate  →  data/curated/*.jsonl   (Sample: x + ground-truth y)
infer   →  runs/*.jsonl           (Prediction: ŷ + confidence)
evaluate→  join on sample_id → accuracy / ECE / AUROC per task
```

## Install

```bash
python3 -m pip install -r requirements.txt      # or: pip install -e .
```

Requires Python 3.9+. The scripts add `src/` to the path, so an editable install
is optional.

## Credentials (API keys)

The scripts auto-load a `.env` file at the repo root — no manual `export` needed
(a real environment variable still overrides the file). Copy the template and
fill in your keys:

```bash
cp .env.example .env
# then edit .env:
#   GEMINI_API_KEY=...   # Google AI Studio key, https://aistudio.google.com/apikey
#   HF_TOKEN=hf_...      # HuggingFace token (only needed to fetch real images)
```

`.env` is gitignored, so your keys are never committed. You only need:

- **`GEMINI_API_KEY`** — for `--backend gemini`. Free at
  <https://aistudio.google.com/apikey>. (Not needed for the offline `mock`
  backend.)
- **`HF_TOKEN`** — only for `scripts/fetch_images.py` (downloading the gated
  DriveBench/DriveLM images). Create at
  <https://huggingface.co/settings/tokens>; a classic **Read** token works, or a
  fine-grained token with *"Read access to public gated repos"* enabled. Also
  click **Agree to access** once at
  <https://huggingface.co/datasets/OpenDriveLab/DriveLM>.

## Quickstart (offline, no API key)

A synthetic DriveBench-format fixture lets you run the whole pipeline with a
deterministic **mock** backend — useful for validating plumbing before spending
Gemini quota.

```bash
python3 scripts/make_sample_data.py                       # tiny fixture + images
python3 scripts/curate.py --dataset drivebench \
    --qa-file data/sample/drivebench_sample.json \
    --data-root data/sample --split clean --formats mcq \
    --out data/curated/drive_v1.jsonl
python3 scripts/infer.py --data data/curated/drive_v1.jsonl \
    --backend mock --strategy verbal_confidence --out runs/mock_vc.jsonl
python3 scripts/evaluate.py --curated data/curated/drive_v1.jsonl \
    --pred runs/mock_vc.jsonl
```

## Running on Gemini (Google AI Studio free tier)

Set `GEMINI_API_KEY` in `.env` (see [Credentials](#credentials-api-keys)), then
just swap the backend:

```bash
python3 scripts/infer.py --data data/curated/drive_v1.jsonl \
    --backend gemini --model gemini-2.5-flash-lite \
    --strategy verbal_confidence --concurrency 4 --out runs/gemini_vc.jsonl
```

Model note: free-tier quota varies by model/region — `gemini-2.5-flash-lite` is
the most reliably-free multimodal model; `gemini-2.0-flash` / `gemini-2.5-flash`
may return `429 ... limit: 0` on some accounts. Probe what your key allows:

```bash
python3 -c "import _bootstrap,os; from google import genai; \
print(genai.Client(api_key=os.environ['GEMINI_API_KEY']).models.generate_content(\
model='gemini-2.5-flash-lite', contents='ok').text)"
```

Keep `--concurrency` low (≈4) for rate limits; runs resume automatically if
interrupted (re-run the same command — completed `sample_id`s are skipped).

## Running on real DriveBench

DriveBench ships in two pieces: **QA annotations** (free) and **nuScenes images**
(gated, one-click).

**1. Annotations** — clean split, no login (the 16 corruption settings are the
other `*.json` files in the same repo):

```bash
mkdir -p data/raw/drivebench
curl -L -o data/raw/drivebench/drivebench-test.json \
  https://huggingface.co/datasets/drive-bench/arena/resolve/main/drivebench-test.json
```

1,461 QA over 200 keyframes (perception 400 / planning 600 / prediction 261 /
behavior 200). All open-ended **except behavior**, which is multiple-choice with
single-letter answers (clean exact-match for ECE/AUROC).

**2. Images** — the matching nuScenes frames are on HuggingFace as
`OpenDriveLab/DriveLM` (gated: auto). Set `HF_TOKEN` in `.env` and accept the gate
(see [Credentials](#credentials-api-keys)). Then fetch only your subset's images
(range requests — no 705 MB download). Note the arena frames live in the **train**
split:

```bash
python3 scripts/fetch_images.py --qa-file data/raw/drivebench/drivebench-test.json \
    --frames 15 --split train --out-root data/raw/drivebench
```

**3. Curate + run** (subset of the clean split, real images):

```bash
python3 scripts/curate.py --dataset drivebench \
    --qa-file data/raw/drivebench/drivebench-test.json \
    --data-root data/raw/drivebench --split clean --per-task 10 \
    --out data/curated/drivebench_real.jsonl
python3 scripts/infer.py --data data/curated/drivebench_real.jsonl \
    --backend gemini --model gemini-2.5-flash-lite \
    --strategy verbal_confidence --concurrency 4 --out runs/gemini_real.jsonl
python3 scripts/evaluate.py --curated data/curated/drivebench_real.jsonl \
    --pred runs/gemini_real.jsonl   # exact-match metrics are meaningful for behavior
```

The adapter (`src/avbench/curation/adapters/drivebench.py`) maps the arena schema
(`question_type`, `image_path`) and resolves images by basename, so layout
differences don't matter. `--split all` keeps every corruption setting; named
splits (download the matching `*.json`, e.g. `fog.json`) become the OOD axis for
O2-KR3. Open-ended perception/prediction/planning answers need a BLEU/GPT-score
judge for automatic correctness — eyeball them for now.

## Config files

Both `curate.py` and `infer.py` accept `--config <yaml>`; CLI flags override
config values, which override built-in defaults. The canonical setups live in
`configs/`:

```bash
python3 scripts/curate.py --config configs/datasets/drivebench.yaml
python3 scripts/infer.py  --config configs/runs/gemini_verbal_conf.yaml
# override anything ad hoc:
python3 scripts/curate.py --config configs/datasets/drivebench.yaml --per-task 5
```

## What's implemented

| Layer | Pieces |
|-------|--------|
| Schema | `Sample`, `Prediction` (Pydantic) — the contract everything shares |
| Curation | `DatasetAdapter` ABC + registry, **DriveBench** adapter, nuScenes sensor-path linking, task-stratified sampling |
| Inference | `VLMClient` ABC, **Gemini** + **mock** backends, async runner (bounded concurrency, resume, error capture) |
| Strategies | `direct`, `verbal_confidence`, `consistency` (self-consistency agreement) |
| Eval | accuracy, ECE, AUROC — overall and per `task_type` |

## Extending

- **New dataset** → add `src/avbench/curation/adapters/<name>.py` with
  `@register("<name>")`, yield `Sample`s. Nothing downstream changes.
- **New confidence method** (CoT, abstention/self-reflection, VL-Uncertainty…) →
  add `src/avbench/inference/strategies/<name>.py` with `@register("<name>")`.
- **Local model** (vLLM, O2/O3) → add a backend implementing `VLMClient`
  (OpenAI-compatible). Token-logprob strategies light up automatically when the
  backend populates `GenResult.avg_logprob` (Gemini free tier usually doesn't).

## Status / known gaps

- MCQ correctness is exact-match. Open-ended QA/CAP need BLEU / GPT-score scorers
  (deliberately deferred — that's why we lead with MCQ).
- Token-probability confidence depends on backend logprob support.
- The bundled fixture is synthetic (solid-color images, toy MCQs) purely to
  exercise the pipeline — not real driving data.
```
