# API reference (0.1.0)

Everything listed here is exported from `adaptive_harness`. These signatures were taken from the code with
`inspect` on 2026-11-11. Anything not listed (modules with a leading underscore, `_`-prefixed methods) is internal
and may change.

## `AdaptiveHarness`

```python
AdaptiveHarness(provider: str, model: str, *, api_key: str | None = None, base_url: str | None = None,
                config: HarnessConfig | None = None, adapter: ProviderAdapter | None = None,
                strategies: StrategyRegistry | None = None, policy_seed: int | None = None,
                aux_adapters: Mapping[str, ProviderAdapter] | None = None, **provider_options)
```

**Arguments:**
- `provider`: one of `openai`, `anthropic`, `gemini`, `groq`, `openrouter`, `deepseek`, `ollama`, `ollama_cloud`,
  `vllm`, `lmstudio`, `openai_compatible` (with `base_url`), or `mock` (for tests).
- `api_key`: optional; otherwise the provider's environment variable is read (see `env.example`).
- `config`: when you pass one, it is used as given. When `config` is omitted, `ADAPTIVE_HARNESS_*` environment
  settings apply.
- `provider_options`: passed to the adapter (for example `extra_body=` for OpenAI-compatible endpoints).

| Method | What it does |
|---|---|
| `generate(prompt, *, system=None, check=None, background=False, extra=None, **settings)` | Send a prompt through the full loop (analyse, choose, transform, call, record, evaluate, learn). Returns `HarnessResponse` |
| `chat(messages, *, check=None, background=False, extra=None, **settings)` | Same, for a message list (dicts or `Message`), including tool turns |
| `agenerate(...)`, `achat(...)` | Async versions; planning and checks run off the event loop |
| `stream(prompt, *, system=None, check=None, background=False, extra=None, **settings)` | Yields `StreamEvent`s; the last one carries the result. The check runs on the full text |
| `preview(prompt, *, system=None, extra=None, **settings)` | What `generate` would send now, without calling the model. Returns `Preview` |
| `feedback(response_id, *, success=None, score=None)` | Report the real outcome of an earlier response, once. It counts as evidence for the profile that served it |
| `profile()` | The current model profile and its evidence, or `None` before the first call |
| `model_info()` | Ask the provider for model metadata; facts are stored with their provenance |
| `flush(timeout=None)` | Wait for background checks (and a running automatic calibration) |
| `close(timeout=30.0)` | Cancel background calibration, wait for pending checks, release resources. The harness is also a context manager |
| `run_trial(prompt, *, strategy, check, experiment_id, task_ref, **settings)` | One experiment call with a fixed strategy (used by calibration and research) |

**`settings`:** `max_tokens`, `temperature`, `top_p`, `stop`, `seed`. A setting the provider cannot accept is
recorded in `result.dropped_settings`.

**`extra`:** provider-native options sent unchanged, for example `tools`, `tool_choice` or `response_format`.

**`check`:** a function `text -> bool | float in [0, 1]`, or an object with `evaluate(text) -> EvaluationResult`
(see "Evaluators" below). Only checks and `feedback()` change what the harness learns. A crashing check is an
`evaluation_error`, never a model failure.

## Results

- **`HarnessResponse`:** `id`, `text`, `result` (`CompletionResult`) and `trace` (`Trace`). Properties: `usage`,
  `cost_usd`, `tool_calls`. `str(response)` is the text.
- **`Trace`:** `strategy_id`, `strategy_version`, `reason`, `optimized`, `category`, `confidence`, `candidates`,
  `explored`, `p_better`, `evaluation_status`, `notes`.
- **`Preview`:** `trace` and `messages` (exactly what would be sent).
- **`Message`:** `role` (`system`, `user`, `assistant` or `tool`), `content`, `tool_calls`, `tool_call_id`,
  `name`.
- **`ToolCall`:** `name`, `arguments` (JSON text), `id`, `signature` (an opaque provider token; send it back
  unchanged).

## Configuration (all frozen dataclasses)

| Class | Fields and defaults |
|---|---|
| `HarnessConfig` | `optimize=True`, `budget`, `persistence`, `http`, `policy`, `rewriter=None`, `calibration`, `llm_classifier=None`, `log_content=False`; also `HarnessConfig.from_env()`, `from_toml(path)`, `from_dict(d)` |
| `BudgetConfig` | `total_usd=0.0`, `daily_usd=None`. Caps the harness's own spending; your own calls are recorded, never blocked |
| `PersistenceConfig` | `enabled=True`, `path=None` (per-user data directory), `prompt_storage="none"` (`"none"`, `"redacted"` or `"full"`) |
| `HTTPConfig` | `timeout_s=120.0`, `max_attempts=3`, `max_backoff_s=30.0`, `max_total_wait_s=90.0` |
| `PolicyConfig` | `exploration_rate=0.1`, `exploration_target=60`, `min_gain=0.02`, `confidence=0.9`, `min_trials=30`, `cost_weight=0.1`, `half_life_days=90.0`, `alias_half_life_days=21.0`, `drift_window=30`, `drift_reference=120`, `drift_z=4.0`, `prior_evidence=True` |
| `CalibrationConfig` | `auto=None` (on for real local providers only), `max_calls=60`, `max_cost_per_call_usd=0.0` (must be > 0 for hosted providers), `max_tokens=600`, `categories=("math", "structured_output")`, `min_interval_s=3600.0` |
| `RewriterConfig` | `rewriter_provider`, `rewriter_model`, `judge_provider`, `judge_model`, `max_cost_per_call_usd=0.001`, `max_tokens=2000`; the judge must be a different model |
| `ClassifierConfig` | `provider`, `model`, `max_cost_per_call_usd=0.001`, `max_tokens=1000` |

## Evaluators (`adaptive_harness.evaluation.builtin`)

`PythonSyntax`, `JsonOutput`, `JsonFields`, `ReferenceAnswer`, `UnitTests` (runs in the Docker sandbox) and
`BulletConstraints`. All return `EvaluationResult` with an `EvalStatus`: `success`, `failure`, `partial`,
`evaluation_error`, `not_applicable` or `no_evaluation`. Only the first three count as evidence.

## Strategies

Built-in strategies: `identity`, `concise_directive`, `role_framing`, `stepwise_plan`, `output_schema`,
`security_generic`, `structured_sections`, `requirements_checklist` and `minimal_context`. `llm_rewrite` is
opt-in through `RewriterConfig`.

To add your own, subclass `adaptive_harness.strategies.Strategy` (with a `StrategyInfo` and `transform()`), and
pass a `StrategyRegistry` that contains it. Every transformation still goes through the preservation guard.

## Exceptions

All derive from `HarnessError`:
- `ConfigurationError`, `BudgetExceededError`, `StoreError`;
- `ProviderError`, with its subclasses `AuthenticationError`, `RateLimitError` (retried), `QuotaExceededError`
  (not retried), `InvalidRequestError`, `ContextLengthError`, `ProviderTimeoutError`, `ProviderUnavailableError`,
  `MalformedResponseError`.
