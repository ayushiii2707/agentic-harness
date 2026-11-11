# adaptive-prompt-harness

A model-agnostic harness that sits between your code and the LLM you call. It learns, with measured evidence,
how that specific model responds to different prompting strategies for different kinds of tasks, and it
transforms your prompts automatically. By default it sends your original prompt unchanged until the evidence
says a change helps.

> **Status: pre-alpha.** The decision loop works end to end:
> - tasks are classified, and strategies are chosen from stored evidence;
> - outcomes you report update that evidence, and the next choice uses it;
> - evidence decays, borrows weakly from related cells, and drift marks a profile stale.
>
> Also built:
> - budgeted calibration, which can also run automatically;
> - a Docker sandbox for code checks;
> - background checks, streaming checks and tool pass-through;
> - opt-in LLM helpers (a rewriter with an independent judge, and a classifier).
>
> In progress: the pre-registered experiment ([experiments/](experiments/README.md)), whose results are not final.
> The package has not been published; until it is, install it from source (below). See
> [docs/DESIGN.md](docs/DESIGN.md) and [docs/DEVLOG.md](docs/DEVLOG.md).

## Install

Python 3.10–3.13. The package is not on PyPI yet; install it from source:

```bash
pip install "git+https://github.com/ayushiii2707/agentic-harness@feat/foundation-and-providers"
```

This needs read access to the repository if it is private. The development branch is currently the only branch.

For development:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The code-test evaluator (`UnitTests`) needs Docker and the pinned image
(`docker pull python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159`). Nothing
else needs Docker.

## Quickstart

```python
from adaptive_harness import AdaptiveHarness

h = AdaptiveHarness(provider="ollama", model="qwen2.5-coder:7b")   # local, free, no key
r = h.generate("Explain binary search and implement it in Python.")
print(r)          # the model's answer
print(r.trace)    # category, strategy used, and why
```

### How it learns

The harness only learns from **real outcomes**: a check you supply, or feedback you report. Without them it never
counts an answer as a success or failure, and it keeps sending your original prompt.

By default:
- **Checked calls.** On 10% of calls with a check, it tries another strategy, but only while the comparison still
  lacks data. It stops once every candidate has about 60 outcomes.
- **Local models.** A new local model (Ollama, vLLM, LM Studio) is calibrated automatically in the background, for
  free.
- **Hosted providers.** Calibration costs money there, so it only runs if you allow it, with a per-call cap and a
  budget.

```python
from adaptive_harness import AdaptiveHarness

h = AdaptiveHarness(provider="groq", model="openai/gpt-oss-20b")

r = h.generate("What is 17 * 23? End with the number only.", check=lambda text: text.strip().endswith("391"))
r = h.generate("Summarise this release note ...")
h.feedback(r.id, success=True)    # or score=0.7

h.profile()["evidence"]           # per category and strategy: outcomes seen so far
```

A strategy replaces your original prompt only after both have at least 30 evaluated outcomes and the evidence
says it is better by a margin, with high posterior probability (DESIGN §6.4). Evidence is kept per model version,
in a local SQLite file.

### More

```python
r = h.generate(prompt, check=slow_check, background=True)   # returns now; the check runs on a worker thread
h.flush()                                                     # wait for background checks

for event in h.stream(prompt, check=my_check):                # streaming; the check runs on the full text
    print(event.delta, end="")

r = h.chat(messages, extra={"tools": [...]})   # provider-native tool definitions pass through unchanged
call = r.tool_calls[0]                           # normalised: name, arguments (JSON), id
messages += [{"role": "assistant", "content": r.text, "tool_calls": [call]},
             {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": '{"temp_c": 4}'}]
r = h.chat(messages, extra={"tools": [...]})   # the model answers using the tool result
```

Configuration you pass in code always wins. Environment variables and a `.env` file apply only when you pass no
config, and `.env` is read only inside your project (up to the folder with `.git` or `pyproject.toml`).

Opt-in features:
- `CalibrationConfig(auto=True, max_cost_per_call_usd=..., ...)` with a budget also calibrates hosted models.
  `auto=False` turns automatic calibration off everywhere.
- `ClassifierConfig(provider, model)` lets an LLM label prompts that the built-in classifier calls `general`.
- `RewriterConfig(...)` enables the LLM rewriter. An independent judge must confirm every rewrite.

API keys are read from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
`GROQ_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY`) or from a local `.env` file. Keys are never logged or
stored. See [env.example](env.example).

## Supported providers

| Provider | Adapter | Notes |
|---|---|---|
| OpenAI, Groq, OpenRouter, DeepSeek | `openai_compatible` presets | |
| Ollama, vLLM, LM Studio (local) | `openai_compatible` presets | No key needed |
| Anthropic | native Messages API | |
| Google Gemini | native Generative Language API | |

Which adapters have been tested against the live service, and which only against recorded responses, is listed
in [docs/PROVIDERS.md](docs/PROVIDERS.md).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `trace.strategy_id` is always `identity` | Expected until there is evidence. The harness learns only from `check=` or `feedback()`, and needs 30 outcomes per strategy before it switches. Use `h.preview(prompt).trace.reason` to see why |
| `ConfigurationError: ... max_cost_per_call_usd > 0` | Paid calibration or helper models need a per-call cap and a budget (`BudgetConfig(total_usd=...)`); a $0 cap would let spending go untracked |
| `QuotaExceededError` | The provider account has no credit or quota left. It is not retried; add credit, or use a free or local model |
| `RateLimitError` after retries | A temporary limit. Wait and retry; calibration and experiments resume where they stopped |
| Calls to a local Ollama time out | The first load of a large model can take minutes; raise `HTTPConfig(timeout_s=...)` |
| `UnitTests` results are `evaluation_error` | Docker is not running or the pinned image is missing; code is never run outside the sandbox |
| A setting seems ignored | Check `response.result.dropped_settings`: providers differ in which settings they accept |

## Documents

- [API reference](docs/API.md), [Design](docs/DESIGN.md), [Related work](docs/RELATED_WORK.md),
  [Decisions](docs/DECISIONS.md), [Development log](docs/DEVLOG.md), [Data flows](docs/DATA_FLOWS.md),
  [Providers](docs/PROVIDERS.md), [Licences](docs/LICENSES.md), [Release checklist](docs/RELEASE_CHECKLIST.md),
  [Changelog](CHANGELOG.md), [Contributing](CONTRIBUTING.md)

## Licence

Apache-2.0. See [LICENSE](LICENSE).
