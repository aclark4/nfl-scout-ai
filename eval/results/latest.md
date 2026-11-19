## Evaluation results

Model: `claude-haiku-5-5` · Run: 2026-11-19T09:21:55 · 18 questions per mode

| Mode | Accuracy | Correct tool used | Median latency | Avg tokens / question |
|---|---|---|---|---|
| rag_only | 16/18 (89%) | n/a | 1.5 s | 1,733 |
| rag_tools | 17/18 (94%) | 13/13 (100%) | 2.3 s | 5,823 |

**Accuracy by category**

| Category | rag_only | rag_tools |
|---|---|---|
| lookup | 5/5 (100%) | 5/5 (100%) |
| comparison | 2/2 (100%) | 2/2 (100%) |
| ranking | 4/6 (67%) | 5/6 (83%) |
| methodology | 2/2 (100%) | 2/2 (100%) |
| out_of_scope | 3/3 (100%) | 3/3 (100%) |

**Failures**

- `rag_only` / `sack-avoidance`: contained ['Andy Dalton']
- `rag_only` / `rank-deepest-adot`: missing ['Anthony Richardson', '12.2']
- `rag_tools` / `sack-avoidance`: contained ['Andy Dalton']
