# PYP vs DP Clustering Validation - Training Log

## Experiment 1: EASY MODE - A10 GPU, 100K Samples

**Date:** 2026-01-19
**Hardware:** NVIDIA A10 (24GB VRAM)
**Instance:** Vast.AI

### Configuration
```
n_samples: 100,000
n_features: 60
n_true_clusters: 30
zipf_exponent: 1.5
cluster_separation: 3.0  (EASY - well separated)
elliptical: False (spherical clusters)
max_components: 100
```

### Results

| Metric | DP (sklearn) | PYP (Pyro/GPU) |
|--------|--------------|----------------|
| K inferred | 30 | 3 |
| NMI | 1.0000 | 0.2856 |
| ARI | 1.0000 | 0.0970 |
| Time (s) | 119.9 | 117.4 |

**Conclusion:** DP perfect, PYP collapsed. Easy problem favors DP.

---

## Experiment 2: HARD MODE - Elliptical Overlapping Clusters

**Date:** 2026-01-19
**Hardware:**
- DP: Alienware (Intel Core Ultra 9 285K)
- PYP: NVIDIA A10 (Vast.AI)

### Configuration
```
n_samples: 50,000
n_features: 60
n_true_clusters: 30
zipf_exponent: 2.0  (steeper power-law, more niche clusters)
cluster_separation: [0.5, 1.0, 2.0]  (HARD - overlapping)
elliptical: True (random covariance matrices)
max_components: 100
```

### Data Generation
```
Cluster sizes (Top 5):    [5319, 5319, 3191, 3191, 2127]
Cluster sizes (Bottom 5): [1063, 1063, 1063, 1063, 1063]
```

### DP Results (Alienware CPU, cov=full)

| Separation | K_inferred | K_true | NMI | ARI | Time (s) |
|------------|------------|--------|-----|-----|----------|
| 0.5 | 100 | 30 | 0.8883 | 0.7761 | 4197 |
| 1.0 | 100 | 30 | 0.8903 | 0.7646 | 3841 |
| 2.0 | 100 | 30 | 0.8971 | 0.7962 | 5303 |

**DP Analysis:** Over-segmentation! DP finds 100 clusters instead of 30.
High NMI (~0.89) but wrong cluster count.

### PYP Results (A10 GPU, TUNED settings)

| Separation | K_inferred | K_true | NMI | ARI | Time (s) |
|------------|------------|--------|-----|-----|----------|
| 0.5 | 1 | 30 | 0.0000 | 0.0000 | 149 |

**PYP Analysis:** Complete collapse! PYP merges everything into 1 cluster.
ELBO loss didn't converge properly.

### PYP Training Loss (separation=0.5)
```
Step 400/2000:  ELBO loss = 4582116.29
Step 800/2000:  ELBO loss = 4570019.99
Step 1200/2000: ELBO loss = 4567689.40
Step 1600/2000: ELBO loss = 4648851.22  (increased!)
Step 2000/2000: ELBO loss = 4599367.18
```
Loss oscillating - optimization unstable.

---

## Summary: Head-to-Head Comparison

### EASY MODE (separation=3.0, spherical)

| Method | K | NMI | Winner |
|--------|---|-----|--------|
| DP | 30 ✅ | 1.0000 | **DP** |
| PYP | 3 | 0.2856 | |

### HARD MODE (separation=0.5, elliptical)

| Method | K | NMI | Winner |
|--------|---|-----|--------|
| DP | 100 (over) | 0.8883 | **DP** |
| PYP | 1 (collapsed) | 0.0000 | |

---

## Final Conclusions

### Claims Tested

| Claim | Result |
|-------|--------|
| "PYP beats DP by 17-22% on power-law data" | ❌ NOT VALIDATED |
| "DP underestimates cluster count" | ❌ OPPOSITE - DP over-estimates |
| "PYP better for niche clusters" | ❌ PYP collapsed to 1 cluster |

### Why PYP Failed

1. **Variational inference limitations** - Mean-field approximation too restrictive
2. **Elliptical clusters** - PYP uses diagonal covariance, can't model ellipses
3. **Optimization instability** - ELBO loss oscillated, didn't converge
4. **High-dimensional curse** - 60 features is challenging for stick-breaking

### Why DP Over-Segmented

1. **Full covariance estimation** - With overlapping ellipses, splits uncertain regions
2. **Conservative merging** - DP prior favors more clusters with concentration=1.0
3. **Actually reasonable** - 100 clusters with NMI=0.89 means mostly correct structure

---

## RECOMMENDATION FOR HIMARI

### Use sklearn's Dirichlet Process (BayesianGaussianMixture)

**Reasons:**
1. ✅ Robust - works on both easy and hard problems
2. ✅ Achieves NMI=0.89-1.0 across conditions
3. ✅ No GPU required - CPU-only implementation
4. ✅ Mature library - well-tested, production-ready
5. ✅ Simple API - few hyperparameters to tune

**Configuration:**
```python
from sklearn.mixture import BayesianGaussianMixture

model = BayesianGaussianMixture(
    n_components=100,  # Upper bound, will prune
    covariance_type='full',  # For elliptical clusters
    weight_concentration_prior_type='dirichlet_process',
    weight_concentration_prior=1.0,
    max_iter=500,
    n_init=3,
)
```

**Post-processing:**
- If over-segmentation is a problem, merge clusters with high similarity
- Or reduce `n_components` to constrain cluster count

### DO NOT Use Pyro PYP (for now)

**Reasons:**
1. ❌ Collapsed to 1-3 clusters on all tests
2. ❌ Requires extensive hyperparameter tuning
3. ❌ Optimization unstable (loss oscillation)
4. ❌ Mean-field variational approximation too restrictive

**If revisiting PYP later:**
- Use full covariance matrices (not diagonal)
- Try collapsed Gibbs sampling instead of SVI
- Consider NumPyro for better MCMC
- Tune discount parameter more carefully (try 0.1-0.5 range)
