-- 0001: initial evidence store schema (docs/DESIGN.md section 8).
-- Forward-only. Never edit a released migration; add a new numbered file instead.

CREATE TABLE model (
  id               INTEGER PRIMARY KEY,
  provider         TEXT NOT NULL,
  model_id         TEXT NOT NULL,
  resolved_version TEXT,
  deployment       TEXT NOT NULL,              -- hosted | local:<host>
  fingerprint      TEXT,
  alias            INTEGER NOT NULL DEFAULT 0,
  first_seen       REAL NOT NULL,
  last_seen        REAL NOT NULL,
  UNIQUE (provider, model_id, resolved_version, deployment)
);

CREATE TABLE model_fact (
  id          INTEGER PRIMARY KEY,
  model_ref   INTEGER NOT NULL REFERENCES model(id),
  key         TEXT NOT NULL,
  value_json  TEXT NOT NULL,
  provenance  TEXT NOT NULL CHECK (provenance IN ('verified','provider_reported','inferred','observed')),
  observed_at REAL NOT NULL
);
CREATE INDEX ix_model_fact ON model_fact(model_ref, key);

CREATE TABLE strategy (
  id            TEXT NOT NULL,
  version       TEXT NOT NULL,
  content_hash  TEXT NOT NULL,
  kind          TEXT NOT NULL CHECK (kind IN ('identity','additive','structural','rewriting')),
  metadata_json TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY (id, version)
);

CREATE TABLE task_category (
  id               TEXT PRIMARY KEY,
  parent_id        TEXT REFERENCES task_category(id),
  taxonomy_version TEXT NOT NULL
);

CREATE TABLE profile (
  id                 INTEGER PRIMARY KEY,
  model_ref          INTEGER NOT NULL REFERENCES model(id),
  profile_version    INTEGER NOT NULL,
  status             TEXT NOT NULL CHECK (status IN ('cold','calibrating','active','stale')),
  created_at         REAL NOT NULL,
  last_calibrated_at REAL,
  notes              TEXT,
  UNIQUE (model_ref, profile_version)
);

CREATE TABLE prior_evidence (
  id                 INTEGER PRIMARY KEY,
  citation           TEXT NOT NULL,
  model_id_as_tested TEXT NOT NULL,
  host               TEXT,
  category           TEXT NOT NULL,
  strategy           TEXT NOT NULL,
  metric             TEXT NOT NULL,
  effect             REAL,
  ci_low             REAL,
  ci_high            REAL,
  verdict            TEXT,
  evidence_date      TEXT NOT NULL,
  weight             REAL NOT NULL DEFAULT 1.0
);

CREATE TABLE arm_stats (
  profile_id        INTEGER NOT NULL REFERENCES profile(id),
  category_id       TEXT NOT NULL,
  strategy_id       TEXT NOT NULL,
  strategy_version  TEXT NOT NULL,
  alpha             REAL NOT NULL DEFAULT 1.0,
  beta              REAL NOT NULL DEFAULT 1.0,
  n_tasks           REAL NOT NULL DEFAULT 0,
  n_trials          INTEGER NOT NULL DEFAULT 0,
  summary_json      TEXT NOT NULL DEFAULT '{}',  -- tokens, cost, latency summaries
  prior_source_ids  TEXT NOT NULL DEFAULT '[]',
  updated_at        REAL NOT NULL,
  PRIMARY KEY (profile_id, category_id, strategy_id, strategy_version)
);

CREATE TABLE experiment (
  id               TEXT PRIMARY KEY,
  kind             TEXT NOT NULL CHECK (kind IN ('online','calibration','research','smoke')),
  config_hash      TEXT NOT NULL,
  suite_version    TEXT,
  software_version TEXT NOT NULL,
  seed             INTEGER,
  budget_json      TEXT NOT NULL DEFAULT '{}',
  status           TEXT NOT NULL DEFAULT 'running',
  started_at       REAL NOT NULL,
  finished_at      REAL
);

CREATE TABLE trial (
  id                 TEXT PRIMARY KEY,
  experiment_id      TEXT REFERENCES experiment(id),
  model_ref          INTEGER REFERENCES model(id),
  provider           TEXT NOT NULL,
  model_requested    TEXT NOT NULL,
  model_reported     TEXT,
  task_ref           TEXT,
  category_id        TEXT,
  strategy_id        TEXT NOT NULL,
  strategy_version   TEXT NOT NULL,
  rep                INTEGER NOT NULL DEFAULT 0,
  settings_json      TEXT NOT NULL DEFAULT '{}',
  prompt_hash        TEXT NOT NULL,
  original_prompt    TEXT,                          -- stored only when privacy allows
  transformed_prompt TEXT,
  response_text      TEXT,
  finish_reason      TEXT,
  input_tokens       INTEGER,
  output_tokens      INTEGER,
  reasoning_tokens   INTEGER,
  cost_usd           REAL,
  cost_is_estimated  INTEGER NOT NULL DEFAULT 0,
  latency_s          REAL,
  retries            INTEGER NOT NULL DEFAULT 0,
  fingerprint_json   TEXT NOT NULL DEFAULT '{}',
  dropped_settings   TEXT NOT NULL DEFAULT '[]',
  error              TEXT,
  status             TEXT NOT NULL CHECK (status IN ('pending','leased','done','failed')),
  simulated          INTEGER NOT NULL DEFAULT 0,
  trace_json         TEXT NOT NULL DEFAULT '{}',
  created_at         REAL NOT NULL,
  finished_at        REAL
);
CREATE INDEX ix_trial_experiment ON trial(experiment_id, status);
CREATE INDEX ix_trial_model ON trial(model_ref, category_id, strategy_id);

CREATE TABLE evaluation (
  id                INTEGER PRIMARY KEY,
  trial_id          TEXT NOT NULL REFERENCES trial(id),
  evaluator_id      TEXT NOT NULL,
  evaluator_version TEXT NOT NULL,
  status            TEXT NOT NULL CHECK (status IN
                      ('success','failure','partial','evaluation_error','not_applicable','no_evaluation')),
  score             REAL,
  details_json      TEXT NOT NULL DEFAULT '{}',
  cost_usd          REAL NOT NULL DEFAULT 0,
  created_at        REAL NOT NULL
);
CREATE INDEX ix_evaluation_trial ON evaluation(trial_id);

CREATE TABLE change_event (
  id          INTEGER PRIMARY KEY,
  profile_id  INTEGER NOT NULL REFERENCES profile(id),
  kind        TEXT NOT NULL CHECK (kind IN ('fingerprint','drift','age')),
  detail_json TEXT NOT NULL DEFAULT '{}',
  at          REAL NOT NULL
);

CREATE TABLE budget_ledger (
  id            INTEGER PRIMARY KEY,
  experiment_id TEXT,
  scope         TEXT NOT NULL,                 -- harness | user
  kind          TEXT NOT NULL CHECK (kind IN ('reserve','commit','release')),
  reservation   TEXT NOT NULL,
  amount_usd    REAL NOT NULL,
  note          TEXT,
  at            REAL NOT NULL
);
CREATE INDEX ix_budget_reservation ON budget_ledger(reservation);
