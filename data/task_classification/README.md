# Task-classification evaluation set

`prompts.jsonl`: 320 prompts, 20 for each of the 16 categories in
`src/adaptive_harness/tasks/taxonomy.json` (15 task categories plus `general`).

- **Provenance:** written by the project author on 2026-11-10 for this purpose. It is not a public benchmark, and
  it is licensed with the repository (Apache-2.0).
- **Split, fixed before any tuning:** within each category, items `-01` to `-12` are the **dev** split (used to
  build the lexicons and tune the thresholds); items `-13` to `-20` are the **test** split, used only for the
  reported numbers. That gives 192 dev and 128 test prompts.
- **Labels:** one category per prompt, the one the author judged primary. Several prompts are genuinely
  ambiguous (e.g. a Flask login API is both `api_development` and `security_sensitive_code`; it is labelled
  `security_sensitive_code`). The taxonomy declares `api_development` and `security_sensitive_code` as children
  of `code_generation`; reports show strict accuracy and a "parent accepted" accuracy.

**Limitation, reported with every result:** the same person wrote the prompts, the labels and the classifier
lexicons, so accuracy here is likely optimistic compared with real user prompts. A second annotator and real
user prompts would make the estimate stronger.
