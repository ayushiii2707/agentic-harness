# Phase 5: task classifier evaluation

Taxonomy version 1; thresholds frozen from dev tuning: min_probability 0.3, min_margin 0.0, unexplained_mass 0.5.

Data: `data/task_classification/prompts.jsonl` (author-written; see its README for the limitation that prompts, labels and lexicons share one author, so these numbers are likely optimistic).

| split | n | accuracy | parent-accepted accuracy | coverage (not `general`) | accuracy when not `general` |
|---|---|---|---|---|---|
| dev (used for tuning) | 192 | 0.938 | 0.943 | 0.943 | 0.934 |
| **test (held out)** | 128 | 0.688 | 0.703 | 0.812 | 0.779 |

## Test: per category

| category | n | precision | recall |
|---|---|---|---|
| api_development | 8 | 1.00 | 0.75 |
| architecture | 8 | 0.86 | 0.75 |
| code_debugging | 8 | 0.71 | 0.62 |
| code_explanation | 8 | 0.67 | 0.75 |
| code_generation | 8 | 0.73 | 1.00 |
| code_refactoring | 8 | 1.00 | 0.62 |
| content_generation | 8 | 0.67 | 0.50 |
| data_analysis | 8 | 0.75 | 0.38 |
| general | 8 | 0.29 | 0.88 |
| math | 8 | 0.67 | 0.25 |
| multi_step | 8 | 1.00 | 0.50 |
| question_answering | 8 | 0.57 | 1.00 |
| research | 8 | 1.00 | 0.75 |
| security_sensitive_code | 8 | 1.00 | 0.75 |
| structured_output | 8 | 0.70 | 0.88 |
| summarisation | 8 | 0.83 | 0.62 |

Of the 40 test errors, 17 fell back to `general` (conservative: only identity and general-safe strategies are then allowed) and 23 assigned a different specific category. A `general` prediction with no matching evidence has confidence 1.0 by construction (all mass is unexplained), which is why that bin looks over-confident.

## Test: most frequent confusions (gold → predicted)

| gold | predicted | count |
|---|---|---|
| content_generation | general | 4 |
| math | general | 3 |
| code_debugging | general | 2 |
| summarisation | general | 2 |
| math | question_answering | 2 |
| multi_step | general | 2 |
| code_debugging | architecture | 1 |
| code_explanation | question_answering | 1 |
| code_explanation | structured_output | 1 |
| code_refactoring | code_generation | 1 |
| code_refactoring | general | 1 |
| code_refactoring | data_analysis | 1 |

## Test: calibration (confidence bin → accuracy)

| confidence | n | accuracy |
|---|---|---|
| 0.2-0.4 | 1 | 1.00 |
| 0.4-0.6 | 25 | 0.56 |
| 0.6-0.8 | 27 | 0.85 |
| 0.8-1.0 | 75 | 0.67 |

## Test: misclassified items

- `db-13`: gold `code_debugging`, predicted `architecture` (confidence 0.40)
- `db-16`: gold `code_debugging`, predicted `general` (confidence 1.00)
- `db-17`: gold `code_debugging`, predicted `general` (confidence 1.00)
- `ex-16`: gold `code_explanation`, predicted `question_answering` (confidence 0.83)
- `ex-17`: gold `code_explanation`, predicted `structured_output` (confidence 0.50)
- `rf-15`: gold `code_refactoring`, predicted `code_generation` (confidence 0.44)
- `rf-19`: gold `code_refactoring`, predicted `general` (confidence 1.00)
- `rf-20`: gold `code_refactoring`, predicted `data_analysis` (confidence 0.44)
- `ar-13`: gold `architecture`, predicted `general` (confidence 1.00)
- `ar-20`: gold `architecture`, predicted `code_explanation` (confidence 0.57)
- `api-15`: gold `api_development`, predicted `code_generation` (confidence 0.50)
- `api-18`: gold `api_development`, predicted `content_generation` (confidence 0.50)
- `da-13`: gold `data_analysis`, predicted `general` (confidence 1.00)
- `da-14`: gold `data_analysis`, predicted `structured_output` (confidence 0.40)
- `da-15`: gold `data_analysis`, predicted `question_answering` (confidence 0.62)
- `da-18`: gold `data_analysis`, predicted `code_debugging` (confidence 0.75)
- `da-19`: gold `data_analysis`, predicted `math` (confidence 0.67)
- `sm-15`: gold `summarisation`, predicted `content_generation` (confidence 0.83)
- `sm-19`: gold `summarisation`, predicted `general` (confidence 1.00)
- `sm-20`: gold `summarisation`, predicted `general` (confidence 1.00)
- `rs-13`: gold `research`, predicted `question_answering` (confidence 0.50)
- `rs-20`: gold `research`, predicted `summarisation` (confidence 0.86)
- `ma-13`: gold `math`, predicted `question_answering` (confidence 0.71)
- `ma-15`: gold `math`, predicted `code_explanation` (confidence 0.80)
- `ma-16`: gold `math`, predicted `question_answering` (confidence 0.83)
- `ma-17`: gold `math`, predicted `general` (confidence 1.00)
- `ma-18`: gold `math`, predicted `general` (confidence 1.00)
- `ma-20`: gold `math`, predicted `general` (confidence 1.00)
- `so-17`: gold `structured_output`, predicted `general` (confidence 1.00)
- `cn-15`: gold `content_generation`, predicted `general` (confidence 1.00)
- `cn-17`: gold `content_generation`, predicted `general` (confidence 1.00)
- `cn-19`: gold `content_generation`, predicted `general` (confidence 1.00)
- `cn-20`: gold `content_generation`, predicted `general` (confidence 1.00)
- `sc-14`: gold `security_sensitive_code`, predicted `structured_output` (confidence 0.83)
- `sc-15`: gold `security_sensitive_code`, predicted `code_generation` (confidence 0.50)
- `ms-13`: gold `multi_step`, predicted `code_debugging` (confidence 0.83)
- `ms-15`: gold `multi_step`, predicted `general` (confidence 1.00)
- `ms-17`: gold `multi_step`, predicted `code_explanation` (confidence 0.57)
- `ms-20`: gold `multi_step`, predicted `general` (confidence 1.00)
- `ge-15`: gold `general`, predicted `question_answering` (confidence 0.83)
