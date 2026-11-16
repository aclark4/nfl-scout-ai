## Evaluation results

Model: `claude-haiku-5-5` · Run: 2026-11-16T12:14:25 · 1 questions per mode

| Mode | Accuracy | Correct tool used | Median latency | Avg tokens / question |
|---|---|---|---|---|
| rag_only | 0/1 (0%) | n/a | 2.4 s | 1,556 |
| rag_tools | 0/1 (0%) | 1/1 (100%) | 2.4 s | 5,557 |

**Accuracy by category**

| Category | rag_only | rag_tools |
|---|---|---|
| ranking | 0/1 (0%) | 0/1 (0%) |

**Failures**

- `rag_only` / `sack-avoidance`: contained ['Andy Dalton']
- `rag_tools` / `sack-avoidance`: contained ['Andy Dalton']
