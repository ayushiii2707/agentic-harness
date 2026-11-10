-- Labels from the opt-in LLM fallback classifier, so a prompt keeps the same category across processes
-- (calibration and serving must look evidence up under the same key; DECISIONS D29, D47).
CREATE TABLE classification_cache (
  prompt_hash      TEXT NOT NULL,
  classifier       TEXT NOT NULL,   -- provider/model
  taxonomy_version TEXT NOT NULL,
  category         TEXT,            -- NULL = the model gave no usable answer; general is kept
  created_at       REAL NOT NULL,
  PRIMARY KEY (prompt_hash, classifier, taxonomy_version)
);
