## Evaluation results

Model: `claude-haiku-5-5` · Run: 2026-11-02T13:32:52 · 17 questions per mode

| Mode | Accuracy | Correct tool used | Median latency | Avg tokens / question |
|---|---|---|---|---|
| rag_only | 16/17 (94%) | n/a | 1.7 s | 1,727 |
| rag_tools | 17/17 (100%) | 12/12 (100%) | 2.5 s | 5,763 |

**Accuracy by category**

| Category | rag_only | rag_tools |
|---|---|---|
| lookup | 5/5 (100%) | 5/5 (100%) |
| comparison | 2/2 (100%) | 2/2 (100%) |
| ranking | 4/5 (80%) | 5/5 (100%) |
| methodology | 2/2 (100%) | 2/2 (100%) |
| out_of_scope | 3/3 (100%) | 3/3 (100%) |

**Failures**

- `rag_only` / `rank-deepest-adot`: missing ['Anthony Richardson', '12.2']
