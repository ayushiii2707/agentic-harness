# adaptive-prompt-harness

A model-agnostic harness that sits between your code and the LLM you call. It learns, with measured evidence,
how that specific model responds to different prompting strategies for different kinds of tasks, and it
transforms your prompts automatically. By default it sends your original prompt unchanged until the evidence
says a change helps.

> **Status: pre-alpha.** Phase 2 (foundation) and Phase 3 (provider adapters) are in progress.
> Strategy selection, calibration and learning are **not implemented yet**: today the harness passes prompts
> through unchanged. See [docs/DESIGN.md](docs/DESIGN.md) for the full design and roadmap.

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
print(r.trace)    # what the harness did (today: identity pass-through)
```

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
