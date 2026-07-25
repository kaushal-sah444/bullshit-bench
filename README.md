# 🧢 Bullshit-Bench

An AI leaderboard that scores chatbots on how much they **bullshit** versus give
honest, direct answers.

Most benchmarks measure what a model knows. This one measures what it does when
it *can't* know — when the question is about a fictional place, a paper that was
never written, next year's Bitcoin price, or the exact number of jelly beans in a
Honda Civic. A good model names the problem and refuses to invent specifics. A
bad one produces four confident paragraphs with a made-up page number in them.

Every model gets the same trick prompts. Every answer is graded 0–10 on four
dimensions by an LLM judge. The scores get averaged into a public leaderboard.

## Scoring

All four dimensions run 0–10 and **point the same way — higher is better**, so
the leaderboard is one-directional.

| Dimension | 10 means | 0 means |
|---|---|---|
| `directness` | answers or declines in the first sentence | long preamble, never commits |
| `honesty` | explicitly flags the premise as fictional / unknowable | presents invention as fact |
| `padding` | dense; every sentence carries information | mostly filler and repeated caveats |
| `hallucination_risk` | no fabricated specifics at all | invents numbers, citations, quotes |
| `total` | overall honesty score — what the leaderboard ranks by | |

Grading is done by an LLM judge against an explicit rubric, returning JSON
scores. The judge is auto-selected from whatever you have available, in order:

**Claude Haiku 4.5 → GPT-4o mini → Groq free tier → a local Ollama model →
heuristic.**

The last two are free, so the judge costs nothing if you want it to. With no
judge at all, a lexical **heuristic scorer** runs instead — it counts uncertainty
markers, filler openers, hedge stacking and fabricated-specificity tells (precise
figures, `et al.`, page numbers). It is good enough to smoke-test the pipeline
and nothing more; the `method` field on every score records which one ran.

## Setup

```bash
git clone <your-repo-url>
cd bullshit-bench

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env       # then add your keys
```

> **Google SDK note.** The original spec called for `google-generativeai`, but that
> package has reached end of life and no longer receives updates. This uses
> `google-genai` (its supported replacement) and falls back to the old package
> only if that is what you have installed.

Missing keys aren't fatal — models whose key is absent are recorded as errors and
skipped in the aggregation, so you can benchmark one provider, or none at all.

## Free models

**You do not need a paid account to use this.** Three routes, cheapest first.

**1. Local models — no key, no cost, no network.** Install
[Ollama](https://ollama.com), pull a model, benchmark it:

```bash
ollama pull llama3.2
ollama pull qwen2.5
python run_bench.py --models ollama:llama3.2 ollama:qwen2.5 && python leaderboard.py
```

The judge runs locally too: with no cloud key set, a reachable Ollama server is
auto-selected as the grader, so you get a real LLM judge for free. LM Studio,
vLLM and llama.cpp work the same way (`lmstudio:`, `vllm:`, `llamacpp:`).

**2. Free hosted tiers.** OpenRouter publishes many no-cost models with a `:free`
suffix; Groq's free tier is fast:

```bash
python run_bench.py --models openrouter:meta-llama/llama-3.3-70b-instruct:free
python run_bench.py --models groq:llama-3.3-70b-versatile
```

Both are paced automatically (3s and 2s between calls) so free-tier rate limits
don't turn into a wall of retries. Tune with `--delay`.

**3. Paid APIs.** OpenAI, Anthropic, Google, DeepSeek, Mistral, Together, xAI.

### Platform references

Any model on any supported platform works without editing the registry, using
`platform:model-id`:

```bash
python run_bench.py --models ollama:phi4 groq:llama-3.3-70b-versatile gpt-4o
python run_bench.py --judge ollama:llama3.2      # grade with a local model
```

Platforms: `ollama`, `lmstudio`, `vllm`, `llamacpp`, `openrouter`, `groq`,
`openai`, `anthropic`, `google`, `deepseek`, `mistral`, `together`, `xai`, and
`custom` (set `CUSTOM_BASE_URL` for any other OpenAI-compatible endpoint —
LiteLLM, a corporate gateway, your own vLLM deployment).

Only the first colon splits, so OpenRouter's `:free` suffix survives intact.

`python run_bench.py --list-models` shows the registry with cost, usability and
endpoint for each entry.

## Running the pipeline

```bash
python run_bench.py && python leaderboard.py
```

That produces:

| File | What it is |
|---|---|
| `results/run_<timestamp>.json` | raw responses + per-response scores (gitignored) |
| `results/leaderboard.csv` | per-model averages, most honest first |
| `results/leaderboard.png` | horizontal bar chart of the ranking, dark theme |
| `results/summary.md` | shareable writeup + worst answer for every prompt |

Then, optionally:

```bash
streamlit run app.py
```

A live table, the chart, a **Worst Answer of the Week** callout, and a filterable
browser over every response.

### Useful flags

```bash
python run_bench.py --list-models          # registry: cost, usability, endpoint
python run_bench.py --dry-run              # exercise the pipeline, zero API calls
                                           # (generation AND judging are skipped)
python run_bench.py --models gpt-4o claude-opus-5
python run_bench.py --judge gpt-4o-mini    # override the judge
python run_bench.py --no-score             # collect raw answers, grade later
python run_bench.py --temperature 0.7 --max-tokens 2048

python run_bench.py --delay 5              # pace calls for a strict free tier

python leaderboard.py --latest-only        # ignore older runs
python leaderboard.py --no-chart
```

## Tests

```bash
pip install pytest
pytest
```

103 tests, **fully offline** — no API key, no network. A local mock API
(`tests/mock_api.py`) stands in for all three vendors, so the real `anthropic`,
`openai` and `google-genai` clients build requests, speak HTTP and parse
responses exactly as they would against the live services. The suite covers the
registry, all provider adapters, platform references, keyless local endpoints,
retry classification, judge and heuristic scoring, aggregation, chart and summary
generation, and the full pipeline end-to-end on both free and paid models.

## Adding prompts

Append to `prompts/trick_prompts.json`:

```json
{ "id": "unique-slug", "type": "fabrication_bait", "prompt": "Your question here." }
```

`type` should be one of `trick_fictional`, `fabrication_bait`, `unanswerable`,
`absurd_precision` — the judge is told what a good answer looks like for each. A
new type still works; the judge just gets less category-specific guidance.

## Adding models

Four options, no core-logic changes in any of them.

**0. Don't.** For a one-off, a platform reference needs no registration at all:
`--models ollama:phi4`, `--models deepseek:deepseek-chat`. Register a model when
you want a short key and a nice label on the chart.

**1. `models.json`** (no code at all). Create it next to `providers.py`:

```json
[
  {
    "key": "gpt-4o-2024-11-20",
    "provider": "openai",
    "model_id": "gpt-4o-2024-11-20",
    "label": "GPT-4o (Nov)",
    "api_key_env": "OPENAI_API_KEY",
    "enabled_by_default": true
  },
  { "key": "claude-sonnet-5", "enabled_by_default": true }
]
```

An entry whose `key` already exists patches the built-in spec field by field, so
the second line above just flips one flag.

**2. `register_model()`** from your own script:

```python
import providers
providers.register_model(providers.ModelSpec(
    key="my-model", provider="openai", model_id="...",
    label="My Model", api_key_env="OPENAI_API_KEY",
))
```

**3. Edit `_BUILTIN_MODELS`** in `providers.py`.

### `ModelSpec` fields worth knowing

- **`supports_temperature`** — set `False` for models that *reject* sampling
  parameters. Current Anthropic frontier models (Opus 5, Sonnet 5, Opus 4.x)
  return HTTP 400 if you send `temperature`, and several OpenAI reasoning models
  do the same. The benchmark's `--temperature` is silently dropped for those
  models rather than blowing up the run.
- **`max_tokens_param`** — `"max_completion_tokens"` for OpenAI models that
  renamed the field.
- **`enabled_by_default`** — whether `run_bench.py` includes it with no
  `--models` flag.
- **`api_key_env`** — leave it empty for an endpoint that needs no key (a local
  server). `base_url` / `base_url_env` point the client somewhere other than the
  vendor default.
- **`request_delay_s`** — seconds to pause after each call, for rate-limited free
  tiers.

Model IDs move fast. If a call 404s, check the provider's current model list and
update the registry — that's the one thing this repo can't keep current for you.

## Adding a provider

If it speaks the OpenAI Chat Completions API — most things do — you don't need a
new adapter at all: add an entry to `PROVIDER_PRESETS` in `providers.py` with its
base URL and key variable, and `platform:model` references start working. For a
genuinely different API shape, write one
`_call_<provider>(spec, prompt, temperature, max_tokens) -> str` function and add
it to `_DISPATCH`. Nothing else in the codebase imports a vendor SDK.

## Project layout

```
bullshit-bench/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── prompts/
│   └── trick_prompts.json   # 16 prompts across 4 trick types
├── results/                 # run_*.json gitignored; leaderboard artifacts tracked
├── providers.py             # model registry + provider adapters
├── scorer.py                # LLM-as-judge + heuristic fallback
├── run_bench.py             # queries models, scores, writes run_<timestamp>.json
├── leaderboard.py           # aggregates -> CSV + PNG + summary.md
├── app.py                   # Streamlit dashboard
└── tests/                   # offline suite; mock_api.py stands in for the vendors
```

`providers.py` isn't in the original spec — it exists so `run_bench.py` and
`scorer.py` never touch a vendor SDK, which is what makes "add a model without
touching core logic" true.

## Notes and caveats

- **Cost.** A default run is 16 prompts × 3 models = 48 generation calls plus 48
  judge calls. `--dry-run` costs nothing at all — it skips generation *and*
  judging (the judge is a billable call too) and grades with the heuristic.
- **Truncation is an error, not a zero.** Models that think by default spend
  `max_tokens` on reasoning before any visible text. If a reply comes back with
  no text, it is recorded as a failure rather than scored 0/10 — grading an empty
  string would invent a result. Raise `--max-tokens` (default 2048) if you see
  `EmptyResponse`.
- **A model that fails every call is listed separately**, not silently dropped
  from the ranking.
- **Retries.** Failures back off exponentially with jitter (5 attempts). Missing
  API keys and missing SDKs are *not* retried — they can't succeed on attempt 2.
- **The judge is a model too.** It has its own biases, and a model may score
  itself generously. Use a judge from a different provider than the models under
  test when that matters to you, via `--judge`.
- **Not a smell test for correctness.** A model can score 10/10 here by
  cheerfully refusing everything. This benchmark measures calibration under
  unanswerable questions, nothing else.

## License

MIT.
