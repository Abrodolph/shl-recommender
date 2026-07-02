# Evaluation Report — Retrieval Recall@10

Query mode: **user-text (deterministic)**  
**Mean Recall@10 = 0.6931**

| Trace | Recall@10 | Hits/Expected | Missed ids |
|-------|----------:|--------------:|------------|
| C1 | 0.67 | 2/3 | opq-universal-competency-report-2-0 |
| C2 | 0.60 | 3/5 | linux-programming-general, smart-interview-live-coding |
| C3 | 0.75 | 3/4 | svar-spoken-english-us-new |
| C4 | 0.80 | 4/5 | basic-statistics-new |
| C5 | 0.40 | 2/5 | global-skills-assessment, global-skills-development-report, salestransformationreport2-0-individualcontributor |
| C6 | 1.00 | 2/2 | — |
| C7 | 0.40 | 2/5 | dependability-and-safety-instrument-dsi, medical-terminology-new, microsoft-word-365-essentials-new |
| C8 | 0.60 | 3/5 | ms-excel-new, ms-word-new |
| C9 | 0.71 | 5/7 | docker-new, sql-new |
| C10 | 1.00 | 2/2 | — |

Recall is measured on retrieval (top-10 catalog ids vs the labeled shortlist mapped to ids). Items the agent adds as defaults (e.g. OPQ32r, Verify G+) that the user never mentions are the main miss source — see `eval/TUNING_LOG.md`.
