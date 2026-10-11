# Robustness analyses (revision)

Scripts for the additional analyses requested during peer review of
*"Modeling the Evolution and Vitality of Temporal Keyword Communities in Cybersecurity Scientific Literature: A Comparative Clustering Study"*.

Place this folder in the repository root, next to `data/`, and run the scripts from inside it:

```bash
pip install -r requirements.txt
cd robustness_analysis
```

All analyses use the released files in `../data/` (the two `*.openai-gpt.jsonl` files). Leiden, Louvain and Infomap are run with fixed random seeds so that every number is reproducible.

## What each script produces

| Order | Command | Produces (revised manuscript) |
|---|---|---|
| 1 | `python norm_eval2.py` | LLM normalization audit — Section 3.2 (i), Table 3 |
| 2 | `python retest.py` | Test–retest and consistency of the LLM output — Section 3.2 (ii)–(iii) |
| 3 | `python run_seeds.py 0 50` | 50-seed Leiden partitions + CVTA (writes `leiden_0_50.pkl`) |
| 4 | `python seed_summary.py` | Seed stability and theme recovery — Section 4.5.2, Table 14; trajectory shape — Section 4.5.3 |
| 5 | `python drift.py` | Tracking-rule sensitivity and cumulative drift — Section 4.4, Table 11, Fig. 18b |
| 6 | `python jstar.py` | Raw vs persistent-vocabulary Jaccard of chain links — Section 3.5.2 |
| 7 | `python variants.py V1 V2 V3` then `python variant_summary.py` | Normalization variants and hub removal — Section 3.2 (iv), Section 4.5.4, Table 4 |
| 8 | `python algos.py` then `python algo_eval.py` and `python algo_summary.py` | Multi-seed runs of five graph-based algorithms, random baseline, split/merge events — Section 4.4, Table 10; selection criteria — Table 12 |
| 9 | `python jstar_base.py` | Persistent-vocabulary Jaccard vs random baseline — Table 10, Fig. 18c |
| 10 | `python coherence.py` | Out-of-year keyword co-occurrence coherence — Section 4.5.1, Table 12 |
| 11 | `python figdiag.py`, `python fig7.py` | Fig. 18 and the revised Fig. 7 |

`common.py` (network construction) and `cvta.py` (CVS, chain construction, CMR, trend slope, zones) are shared modules; `cvta.py` follows the logic of `code/step10_Community_vitality_tracker.py`.

## Notes

- Walktrap and Label Propagation are run as in the original pipeline (default unweighted implementations); Leiden, Louvain and Infomap use edge weights.
- The embedding-based methods (Node2Vec, DeepWalk) are not part of these robustness analyses.
- Approximate run time on a 2-core machine: steps 3 and 8 take about 5–10 minutes each; the rest take a few minutes.
