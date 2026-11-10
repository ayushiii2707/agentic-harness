# Related work and reuse assessment

Status: Phase 1 draft, 2026-11-10. Sources were checked on that date (documentation pages, repositories,
proceedings). Anything marked *unverified* must be re-checked before it is cited in the report.

## 1. The six problem types (from the master brief, Part 17)

| # | Problem | Who solves it | Is it our focus? |
|---|---|---|---|
| 1 | Optimise one prompt (or program) for one task, given a training set and a metric | APE, OPRO, PromptBreeder, TextGrad, DSPy (MIPROv2, SIMBA, GEPA), PromptWizard, ERGO | No. A possible *source* of candidate strategies later |
| 2 | Pick which **model** answers a request | RouteLLM, BaRP, LinUCB multi-LLM selection, commercial routers | No |
| 3 | Pick from a fixed pool of prompt templates under a budget | TRIPLE (best-arm identification), PromptEval, the bandit "prompt design strategy" selection inside OPTS | Partly. This is the algorithm family our calibration uses |
| 4 | Learn **model-specific** prompting preferences | MAPO (model-adaptive prompt rewriting, needs training), PromptBridge (cross-model prompt transfer) | **Yes** |
| 5 | Learn **model × task-category** preferences | No packaged tool found | **Yes** |
| 6 | Keep adapting those preferences as evidence arrives and models change | Routers do this for model choice; nothing found for prompt strategy | **Yes** |

## 2. Systems and papers

### Prompt optimisers (problem 1)

- **APE**, Zhou et al., ICLR 2023 (arXiv 2211.01910). An LLM proposes candidate instructions and the best scorer
  on a dataset is kept. Its adaptive filtering idea (drop weak candidates early) is relevant to calibration cost.
- **OPRO**, Yang et al., ICLR 2024 (arXiv 2309.03409). The optimiser LLM sees past prompts and their scores and
  proposes better ones. Repository `google-deepmind/opro`: Apache-2.0, 12 commits, research code. The README
  itself warns about unexpectedly large API costs.
- **PromptBreeder**, Fernando et al. 2023 (arXiv 2309.16797). Evolves task prompts and mutation prompts. Only
  community implementations exist (e.g. `vaughanlove/PromptBreeder`, MIT, supports Cohere only). One documented
  run costs about $60 and 6.7 hours. That is far outside our per-user budget.
- **TextGrad**, Yuksekgonul et al., published in Nature (arXiv 2406.07496). "Textual gradients" from a stronger
  LLM. Repository `zou-group/textgrad`: MIT, maintained, on PyPI. Optimises variables per task, typically with a
  stronger backward model.
- **DSPy** (Stanford, MIT licence). Optimisers: `BootstrapFewShot*`, `MIPROv2` (instructions + few-shot, Bayesian
  search), `SIMBA`, `GEPA` (reflective Pareto evolution, ICLR 2026, arXiv 2507.19457; outperforms MIPROv2 by more
  than 10 pp on its benchmarks). The DSPy docs say plainly that *"different language models respond differently
  to the same prompt"* and that the same program optimises differently per model. That is the same premise as
  ours. But DSPy needs the developer to write a DSPy program, a training set and a metric, and it optimises
  **offline, per program**. It does not keep a profile per model, and it does not transform arbitrary prompts
  at request time.
- **PromptWizard**, Agarwal et al., ACL Findings 2025 (Microsoft). Self-evolving instructions and examples.
  Reports about 1,730 API calls per task.
- **ERGO**, 2026 (arXiv 2607.20497). Error-to-rule prompt refinement. Reports that prompts transfer from stronger
  to weaker models with no significant loss. This is evidence *against* strong model-specificity for that task
  family, and we must cite it as such.

### Selecting among candidates under a budget (problem 3)

- **TRIPLE**, Shi et al., NeurIPS 2024. Frames prompt selection as fixed-budget **best-arm identification**
  (not regret minimisation) and shows that BAI algorithms beat the earlier selection methods. This directly
  justifies treating calibration as BAI.
- **PromptEval**, NeurIPS 2024. Estimates performance across many templates cheaply with a Rasch-style model.
  Finds **no coherent template preference across LLMs** on MMLU.
- **Bandit-based prompt design strategy selection (OPTS)**, ACL Findings 2025. Thompson sampling over prompt
  design strategies (CoT, role, specificity…) **plus a "do not modify" arm**, inside a prompt optimiser.
  Thompson sampling did best overall. This is the closest algorithmic relative of our policy. It differs in
  scope: it selects strategies during one optimisation run and does not keep persistent per-model or
  per-category profiles.
- **Prompt Duel Optimizer**, ACL Findings 2026. Label-free optimisation with dueling bandits and an LLM judge.
  It also documents *judge circularity* (the optimiser and the judge are the same model). That is a risk we must
  avoid in our model-assisted evaluation.

### Model routing (problem 2)

- **RouteLLM** (LMSYS, Apache-2.0). A router trained on preference data that sends a query to a strong or a weak
  model. Different problem: it chooses the *model*, not the *prompt*.
- **BaRP** (arXiv 2510.07429) and **LinUCB multi-LLM selection** (AAAI 2026). Contextual bandits trained under
  partial feedback, where only the chosen arm's outcome is observed. This is the same feedback structure we
  face online, so their partial-feedback training argument applies to us.

### Model-specific prompting (problem 4)

- **MAPO: model-adaptive prompt optimisation**, Chen et al. 2024 (arXiv 2407.04118). Learns a rewriter for a
  specific model; *unverified detail*: it relies on fine-tuning (SFT/RL). Ours is training-free and works
  through API calls only.
- **PromptBridge**, arXiv 2512.01420 (Dec 2025). Names the problem "**model drifting**": a prompt tuned for one
  model loses performance on another. It calibrates on a small set of alignment tasks and then learns a
  cross-model mapping. That is very close in spirit to our cold-start calibration. Its mechanism (learning a
  prompt-to-prompt mapping between a source and a target model) differs from ours (choosing a strategy per
  model and category from evidence).

### How prompt-sensitive models really are (this bounds what we can gain)

- **Sclar et al., ICLR 2024 (FormatSpread).** Formatting choices change accuracy by up to 76 pp on
  LLaMA-2-13B, unpredictably.
- **ProSA**, EMNLP Findings 2024. Sensitivity varies by dataset and model; larger models are more robust.
- **"Flaw or Artifact?"**, EMNLP 2025. Modern instruction-tuned LLMs are **more robust to prompt templates than
  previously believed**. Much of the measured sensitivity comes from rigid answer matching; it shrinks under
  LLM-as-judge scoring.
- **LeBlanc v4 (our own prior study).** A one-line instruction changed security-test failure by −11.9 to
  +1.5 pp depending on the model.

**Consequence for the design.** The literature does not guarantee that strategy differences on current models
are large. The harness must be **conservative by default**: it sends the original prompt unless the evidence says
otherwise. Phase 12 must be able to report a null result honestly.

### Evaluation frameworks

| Framework | Licence | Role |
|---|---|---|
| Inspect AI (UK AISI) | MIT, very active | Strong candidate for **Phase 12 experiment infrastructure** (datasets, scorers, Docker sandboxes, logs). Too heavy as a runtime dependency of the package |
| lm-evaluation-harness (EleutherAI) | MIT | Raw benchmark runner. Wrong level for request-time evaluation |
| promptfoo | MIT | Node.js prompt testing. Not a Python dependency candidate |
| DeepEval (Confident AI) | Apache-2.0 *(unverified; the search returned a different repo under that name)* | LLM-judge metrics. Possibly an optional extra later |

## 3. What we reuse and what we build

| Need | Decision | Reason |
|---|---|---|
| HTTP calls to providers | `httpx` (BSD-3) | One small dependency; sync and async; easy to mock with `respx` |
| Provider SDKs / LiteLLM | **Not used in core** | Large dependency trees; LiteLLM changes fast. Adapters stay ours, thin and testable |
| DSPy / GEPA / TextGrad | **Not a core dependency**; possible optional "strategy discovery" plug-in after Phase 12 | They solve problem 1. Wrapping one and calling it adaptive intelligence would break rule 17 of the brief |
| Bandit / BAI maths | Implement ourselves (about 200 lines; Beta posteriors, Thompson sampling, successive elimination) | Small, well understood, must be auditable for the thesis |
| Paired statistics for Phase 12 | Implement ourselves, independently of LeBlanc's code (McNemar, BH, cluster bootstrap) | LeBlanc has no licence, so we do not copy code. Re-implementation is about 120 lines |
| Experiment infrastructure | Evaluate Inspect AI in Phase 12 | Mature logging and sandboxing. Keeps experiment code out of the package |
| Sandbox | Docker CLI via `subprocess`, our own in-container runner | No Docker SDK dependency; works with Docker Desktop on macOS and rootless Docker on Linux |

## 4. Dataset licences (affects what the public repository may contain)

| Dataset | Licence | Use |
|---|---|---|
| CodeSecEval (HF `JasonWang1/CodeSecEval`) | **"Currently under review"**. SecEvalBase derives from SecurityEval, which has no explicit licence | **Do not redistribute.** A download script fetches the pinned revision locally; nothing is committed |
| IFEval (verifiable instructions) | Apache-2.0 *(verify in Phase 9)* | Instruction-following category, deterministic checks |
| GSM8K | MIT *(verify)* | Mathematical problem solving, exact match |
| HumanEval / MBPP | MIT / CC-BY-4.0 *(verify)* | Code generation with unit tests |
| Our own suites (JSON-schema tasks, extraction, refactoring) | Ours (Apache-2.0) | Fill categories with no suitable public set |

## 5. What would be new (claims to test, not to assume)

To our knowledge (searches on 2026-11-10), no packaged tool does all of the following. Phase 12 has to show
whether this combination actually helps.
1. Keeps **persistent, versioned strategy profiles per model fingerprint and task category**.
2. Learns them by **budgeted cold-start calibration** plus **online updates from typed evaluation outcomes**.
3. Treats **the original prompt as an explicit default arm** that a strategy must beat with evidence.
4. Transforms prompts at **request time under semantic-preservation constraints**.
5. Detects **model change** and **ages old evidence**.
