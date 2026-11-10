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
> Not built yet: the research experiments and the release work (Phases 12–13). See [docs/DESIGN.md](docs/DESIGN.md) and [docs/DEVLOG.md](docs/DEVLOG.md).

## Install (development)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

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

## Documents

- [Design](docs/DESIGN.md), [Related work](docs/RELATED_WORK.md), [Decisions](docs/DECISIONS.md),
  [Development log](docs/DEVLOG.md), [Data flows](docs/DATA_FLOWS.md)

## Licence

Apache-2.0. See [LICENSE](LICENSE).
