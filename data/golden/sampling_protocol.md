# Sampling and Labelling Protocol: Golden and Dev Evaluation Sets

## 1. Overview and Separation Guarantees
- Golden Evaluation Benchmark (data/golden/golden_set.json): 200 held-out cases.
- Judge Agreement Subset (data/golden/judge_agreement_subset.json): 40 double-scored cases.
- Dev Set (data/dev/dev_set.json): 35 threshold calibration cases.
- Retrieval Corpus strictly excludes all 235 held-out IDs.

## 2. Intent Stratification Breakdown
| Intent Label | Dev Count | Golden Count | Escalation Expectation |
|---|---|---|---|
| `hardware_damage_repair` | 4 | 22 | escalate |
| `warranty_applecare` | 2 | 12 | escalate |
| `purchase_billing` | 4 | 22 | escalate |
| `account_appleid_icloud` | 4 | 22 | auto_handle |
| `battery_performance` | 5 | 30 | auto_handle |
| `software_bug_ios` | 6 | 35 | auto_handle |
| `device_setup_howto` | 3 | 18 | auto_handle |
| `product_availability_preorder` | 3 | 14 | auto_handle |
| `general_feedback_other` | 4 | 25 | auto_handle |
| Total | 35 | 200 | ~38% Escalate / ~62% Auto-handle |
