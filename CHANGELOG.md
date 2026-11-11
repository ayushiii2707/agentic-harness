# Changelog

All notable changes are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/). Before 1.0, minor versions may change the public API.

## [Unreleased] - 0.1.0 (not published)

The first version. Built in Phases 2–12; see `docs/DEVLOG.md` for the full record and `docs/DECISIONS.md` for every
decision.

### Added
- **Request path.**
  - `AdaptiveHarness` with `generate`, `chat`, `agenerate`, `achat` and `stream`.
  - `feedback`, `profile`, `model_info`, `preview`, `flush` and `close`.
- **Providers.**
  - OpenAI-compatible presets: OpenAI, Groq, OpenRouter, DeepSeek, Ollama, Ollama cloud, vLLM, LM Studio, and any
    custom endpoint.
  - Native Anthropic and Gemini adapters.
  - Normalised errors, bounded retries, token usage, cost where reported, and model fingerprints.
- **Tools.** Tool definitions pass through `extra=`. Tool calls are normalised (`ToolCall`), and tool results go
  back to the model (`role="tool"`), including Gemini's thought signatures.
- **Task understanding.**
  - A deterministic classifier over 15 categories, with a `general` fallback.
  - Requirement extraction.
  - An opt-in LLM fallback classifier.
- **Strategies.** Nine versioned, content-hashed built-in strategies, plus an opt-in `llm_rewrite` checked by an
  independent judge.
- **Preservation guard.** Every transformation goes through one guard. A strategy that cannot be applied safely
  sends the original prompt.
- **Learning.**
  - Beta posteriors per (model profile, category, strategy), updated only from checks and feedback.
  - Decay, capped backoff, LeBlanc v4 prior records, a token-cost penalty, drift detection, and profile
    versioning.
- **Calibration.** Budgeted, resumable calibration on built-in suites (math, structured output, instructions,
  code), automatic for local models.
- **Evaluators.** Syntax, JSON, fields, reference answers, exact instruction constraints, and unit tests in a
  locked-down Docker sandbox.
- **Persistence and budget.**
  - SQLite with forward-only migrations.
  - A persistent budget ledger that reserves before every paid harness call.
  - Privacy levels for stored prompts.
- **Research tooling** (`experiments/`, not shipped in the wheel): a pre-registered Phase 12 experiment runner and
  statistics.

### Security
- Keys are masked everywhere; account IDs are masked in provider errors.
- Settings files are read only inside the project, and explicit configuration wins over the environment.
- Generated code runs only in a container with no network, a read-only root and no capabilities.
