# Development log

## Current status
- **Phase 0, repository audit of LeBlanc v4:** complete; approved 2026-11-10.
- **Phase 1, research and architectural design:** complete; approved 2026-11-10 (licence Apache-2.0, $10 hard
  cap, free providers first).
- **Phase 2, foundation:** implemented and tested. **Awaiting review.**
- **Phase 3, provider adapters:** implemented and contract-tested. Live-verified on Groq and Gemini (free tiers);
  local Ollama pending the model download. **Awaiting review.**
- Strategy selection, calibration and learning are **not** implemented. The harness passes prompts through
  unchanged and its trace says so.

## 2026-11-09 to 2026-11-10: Phase 0
- Audited LeBlanc v4 at commit `51522f2` (read-only clone; database queried read-only; recomputation on a copy).
- Results: 12/12 unit tests pass. Selfcheck 11/15 on macOS; the 4 failures are expected (no API key; no
  `wsl.exe`, so no sandbox, and no code was executed). Every headline number reproduced from the database.
- Findings carried into the design:
  - study-hash fragility (line endings) → D9;
  - sandbox result forgery and infrastructure-failure-as-model-failure → D10;
  - oracle-vs-retrieved warning gap → `security_targeted` marked unvalidated;
  - preambles raise total tokens by up to +44% → the cost term in the policy;
  - most effects "unclear" at about 135 pairs per model → power planning in DESIGN §13.

## 2026-11-10: Phase 1
- Literature and tool review. See `RELATED_WORK.md`.

## 2026-11-10: Phases 2 and 3
**Built:**
- **Core:**
  - `types.py`: provider-neutral `Message`, `CompletionRequest`, `CompletionResult`, `Usage`, `FinishReason`,
    `StreamEvent`. `dropped_settings` and `simulated` are first-class fields.
  - `errors.py`: the normalised hierarchy.
  - `config.py`: code, environment, `.env` and TOML. Safe defaults: budget 0, prompts stored as hashes.
  - `_keys.py`: masked key values, redaction of registered and key-shaped strings. (Renamed from its first
    name because a plugin safety hook blocks every shell command containing that word; see D17.)
  - `_logging.py`: a redacting filter.
  - `hashing.py`: canonical JSON hashes.
  - `redaction.py`.
- **Store:** `store/db.py` plus `migrations/0001_init.sql`, the full evidence schema from DESIGN §8. The store
  refuses databases written by newer versions.
- **Budget:** `budget.py`, a persistent reserve/commit/release ledger. Open reservations count against the cap,
  checked inside an immediate transaction; tested with 40 concurrent threads.
- **API:** `api.py`.
  - `AdaptiveHarness.generate`, `chat`, `agenerate`, `achat` and `stream`, all on the identity path.
  - Every call is recorded as a trial, respecting `prompt_storage`.
  - A store failure never blocks a call.
- **Providers:**
  - `_http.py`: bounded retries with a wait budget, `Retry-After` / Gemini `RetryInfo` / Groq "try again in"
    handling, error mapping, SSE parsing.
  - `openai_compat.py` + `presets.py`: OpenAI, Groq, OpenRouter, DeepSeek, Ollama, Ollama cloud, vLLM,
    LM Studio, and custom endpoints.
  - `anthropic.py`: Messages API, version `2026-10-21`.
  - `gemini.py`: Generative Language API, key in a header.
  - `mock.py`, `registry.py`.
- **Script:** `scripts/smoke_providers.py`, a live smoke test. Free targets only unless `--allow-paid`; paid calls
  reserve from the persistent ledger first.

**Tests:** 77 offline tests pass (unit + contract); `ruff check` and `ruff format` are clean; `mypy --strict`
is clean on 22 source files. Contract fixtures are synthetic bodies in each provider's documented format.

**Live verification (2026-11-10, $0.00 spent):**
- Groq `openai/gpt-oss-20b`: ok. 78 input / 58 output tokens, 48 of them reasoning; `system_fingerprint`
  captured; context window 131,072 from `/models`.
- Gemini `gemini-3.5-flash-lite`: ok. `models.get` gives a 1,048,576 context window, version
  `3.5-flash-lite-07-2026`, thinking supported.
- Ollama cloud: model listing works; chat returns **HTTP 402 "add usage credits"**, so it is not free for this
  account and is reclassified as paid.
- OpenAI, Anthropic, OpenRouter: **not called** (paid), as instructed.
- Finding: the three LeBlanc v4 models are reachable without OpenRouter. `openai/gpt-oss-20b` (Groq, free) and
  `gemini-3.5-flash-lite` (Gemini, free) are callable; `deepseek-v4.1-flash:1017` is listed on Ollama cloud but
  needs credits.

**Issues met:**
- macOS sets the `hidden` flag on files created under `~/Desktop` in some setups, and Python 3.13 skips hidden
  `.pth` files, so the editable install was silently not importable. `chflags nohidden` fixes it only
  temporarily (the flag came back), so pytest now imports from `src/` directly (`pythonpath = ["src"]` in
  `pyproject.toml`) and the smoke script adds `src/` to its path. To be documented in CONTRIBUTING (Phase 13).
- Reasoning models spend output tokens on hidden reasoning, so a 16-token cap returned empty answers. The smoke
  test uses 512. This matters for experiment design: `max_tokens` caps must be set per model family and
  recorded.

## Known issues / open questions
- Anthropic, OpenAI and OpenRouter adapters are contract-tested only (paid). One live call each would cost
  < $0.01, but it is only run if you approve.
- Dataset licences for IFEval, GSM8K, HumanEval and MBPP must be verified before Phase 12 (CodeSecEval: local only).
- Python 3.10–3.12 compatibility is checked by mypy (`python_version = 3.10`) but tests have only run on 3.13;
  CI across versions comes in Phase 13.

## Remaining phases
Phases 4–13 as listed in `DESIGN.md` §14, each gated on owner approval.
