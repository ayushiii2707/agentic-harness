# Data flows and privacy

## Where your data goes

| Data | Destination | When |
|---|---|---|
| Your prompt (possibly transformed by the harness) | The provider you configured (e.g. Groq, Gemini, a local Ollama) | Every `generate` / `chat` call |
| Model metadata requests | The same provider (`/models`, `models.get`, `/api/show`) | Only when `model_info()` is called |
| Calibration prompts (our own suites) | The same provider | Only when you run calibration with a budget > 0 (later phase) |
| Prompts sent to a judge or rewriter model | A provider you choose for that role | Only if you enable those optional features (later phases) |

There is **no telemetry**. The harness never contacts any other service. With a local provider (Ollama, vLLM,
LM Studio) nothing leaves your machine.

## What is stored locally

The evidence store is a SQLite file under the harness home (`$ADAPTIVE_HARNESS_HOME`, or the platform default
such as `~/Library/Application Support/adaptive-harness` on macOS).

| `prompt_storage` | Prompts and responses stored as |
|---|---|
| `none` (default) | A hash only. Token counts, costs, latency and trace metadata are kept |
| `redacted` | Text with emails, IPs, long digit runs, password-like assignments and key-shaped strings masked |
| `full` | Full text. Intended for calibration and research on our own task suites |

## Credentials

- Keys are read from the constructor argument, the environment, or a `.env` file (which is git-ignored).
- They are held in a `Secret` wrapper whose `repr`/`str` are masked, are never written to the store, and are
  scrubbed from log records and error messages (registered values and key-shaped patterns).
- Keys travel only in request headers, never in URLs. The Gemini adapter uses `x-goog-api-key` rather than the
  `?key=` query parameter.
