# Provider adapters: status

What "tested" means here:
- **Contract-tested:** request building, response parsing, streaming, error mapping and secret masking are
  covered by offline tests (`tests/contract/`) against **synthetic** response bodies written from the provider's
  documented format. Synthetic fixtures are not recordings of real calls.
- **Live-verified:** a real call through the adapter succeeded (`scripts/smoke_providers.py`), with the date and
  the model.

Paid providers are not called live unless strictly necessary (project rule: free first, $10 hard cap in total).

| Provider | Adapter | Contract-tested | Live-verified | Notes |
|---|---|---|---|---|
| Groq | `openai_compatible` preset `groq` | yes | **yes**, 2026-11-10, `openai/gpt-oss-20b` | Free tier. Reports reasoning tokens and `system_fingerprint` |
| Google Gemini | native `gemini` | yes | **yes**, 2026-11-10, `gemini-3.5-flash-lite` | Free tier. `models.get` exposes context window, version and thinking support |
| Ollama (local) | `openai_compatible` preset `ollama` | yes | **yes**, 2026-11-10, `qwen2.5vl:3b` (the `qwen2.5-coder:7b` download was interrupted by the network and is resuming) | Free, no key. `/api/show` gave family, 3.8B parameters, Q4_K_M quantisation, 128,000 context. A cold model load on an 8 GB Mac exceeded the 120 s default timeout once; raise `HTTPConfig.timeout_s` for large local models |
| Ollama cloud | `openai_compatible` preset `ollama_cloud` | yes | **reached, not completed**: model listing works; chat returns HTTP 402 "add usage credits" | Not free for this account. Mapped to `QuotaExceededError` |
| OpenAI | `openai_compatible` preset `openai` | yes | **reached, not completed** (2026-11-10): models listing works; chat returns HTTP 429 "no credits remaining" | Account has no credit. Mapped to `QuotaExceededError`. Uses `max_completion_tokens` |
| OpenRouter | `openai_compatible` preset `openrouter` | yes | **yes**, 2026-11-10, `openai/gpt-oss-20b`, cost $0.0000066 | Reports per-call cost and serving host (DeepInfra) |
| DeepSeek | `openai_compatible` preset `deepseek` | yes | no (no key) | |
| vLLM, LM Studio | `openai_compatible` presets | yes | no (no local server) | |
| Anthropic | native `anthropic` (Messages API, version `2026-10-21`) | yes | **reached, not completed** (2026-11-10): models listing works; chat returns HTTP 400 "credit balance is too low" | Account has no credit. Mapped to `QuotaExceededError`. `temperature`/`top_p` are not accepted by this API version; they are dropped and recorded in `dropped_settings` |

## Free model availability (checked 2026-11-10, listing calls only)

The three models of the LeBlanc v4 study are reachable on free tiers, so Phase 12 can compare against them
without OpenRouter:
- `openai/gpt-oss-20b` on Groq
- `gemini-3.5-flash-lite` on Gemini
- `deepseek-v4.1-flash:1017` is listed on Ollama cloud, but chat there needs credits

Model availability changes over time. Re-check before every experiment, and record the resolved version that
each trial reports.
