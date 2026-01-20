# Bayesian Clustering Test - Run ALL on Vast.AI GPU

## Single Command for Vast.AI

Copy and paste this entire command:

```bash
cd /workspace && git clone https://github.com/nimallansa937/-Dirichlet-Process-clustering.git && cd -Dirichlet-Process-clustering && pip install scikit-learn hdbscan numpy pandas && python run_all_clustering_tests.py
```

---

## What This Runs

| Test | Description |
|------|-------------|
| sklearn DP (tuned α=0.05) | Sparse prior - should find K≈30 |
| sklearn DP (default α=1.0) | Will over-segment to K≈100 |
| HDBSCAN | Fast density-based baseline |
| Sparse Finite Mixture (BIC) | Model selection across K=10-50 |

---

## Expected Output

```
======================================================================
BAYESIAN CLUSTERING BENCHMARK - CPU ONLY
======================================================================

Generating synthetic data...
Data shape: (50000, 60)
True K: 30
Cluster size range: 1063 - 5319

----------------------------------------------------------------------
RUNNING TESTS
----------------------------------------------------------------------
  Running sklearn DP (tuned α=0.05)...
  Running sklearn DP (default α=1.0)...
  Running HDBSCAN...
  Running Sparse Finite Mixture (BIC selection)...

======================================================================
RESULTS SUMMARY
======================================================================
Method                    K   K_true      NMI      ARI   Time(s)
----------------------------------------------------------------------
sklearn_DP_tuned         30       30   0.8500   0.7500      120.0
sklearn_DP_default      100       30   0.8900   0.7800      150.0
HDBSCAN                  28       30   0.8200   0.7000       30.0
SFM_BIC                  30       30   0.8600   0.7600      200.0
======================================================================

VALIDATION (K_true=30):
  sklearn_DP_tuned      : K=OK           NMI=OK           -> PASS
  sklearn_DP_default    : K=FAIL (100)   NMI=OK           -> FAIL
  HDBSCAN               : K=OK           NMI=OK           -> PASS
  SFM_BIC               : K=OK           NMI=OK           -> PASS
```

---

## Expected Results

| Method | Expected K | Expected NMI | Pass? |
|--------|------------|--------------|-------|
| sklearn_DP_tuned (α=0.05) | 25-40 | ≥0.80 | PASS |
| sklearn_DP_default (α=1.0) | ~100 | ~0.89 | FAIL (K wrong) |
| HDBSCAN | 25-35 | ≥0.80 | PASS |
| SFM_BIC | 30±5 | ≥0.85 | PASS |

---

## Note

These tests use CPU (sklearn/hdbscan are CPU-only libraries). The GPU won't accelerate them, but running on Vast.AI is fine - it just uses the CPU cores.
