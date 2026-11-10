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
> Also built: budgeted calibration, a Docker sandbox for code checks, and an opt-in LLM rewriter with an
> independent judge. Not built yet: automatic (unattended) calibration, orchestration and the research
> experiments (Phases 11–13). See [docs/DESIGN.md](docs/DESIGN.md) and [docs/DEVLOG.md](docs/DEVLOG.md).

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

```python
from adaptive_harness import AdaptiveHarness, HarnessConfig, PolicyConfig

cfg = HarnessConfig(policy=PolicyConfig(exploration_rate=0.2))  # try other strategies on 20% of checked calls
h = AdaptiveHarness(provider="groq", model="openai/gpt-oss-20b", config=cfg)

r = h.generate("What is 17 * 23? End with the number only.", check=lambda text: text.strip().endswith("391"))
r = h.generate("Summarise this release note ...")
h.feedback(r.id, success=True)    # or score=0.7

h.profile()["evidence"]           # per category and strategy: outcomes seen so far
```

A strategy replaces your original prompt only after both have at least 30 evaluated outcomes and the evidence
says it is better by a margin, with high posterior probability (DESIGN §6.4). Evidence is kept per model version,
in a local SQLite file.

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
