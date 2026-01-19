# PYP vs DP Clustering Validation - Training Log

## Experiment 1: A10 GPU, 100K Samples

**Date:** 2026-01-19
**Hardware:** NVIDIA A10 (24GB VRAM)
**Instance:** Vast.AI

### Configuration
```
n_samples: 100,000
n_features: 60
n_true_clusters: 30
zipf_exponent: 1.5
cluster_separation: 3.0
max_components: 100
```

### Environment
```
PyTorch: 2.9.1+cu128
Pyro: 1.9.1
Device: NVIDIA A10
GPU Memory: 23.7 GB
```

### Data Generation
```
Cluster sizes (Top 5):    [21875, 21250, 11875, 8125, 6250]
Cluster sizes (Bottom 5): [625, 625, 625, 625, 625]
```

Power-law distribution confirmed - large clusters dominate, many small niche clusters.

---

## Results

### Dirichlet Process (sklearn)
| Metric | Value |
|--------|-------|
| K inferred | 30 |
| K error | 0 |
| NMI | 1.0000 |
| ARI | 1.0000 |
| Time | 119.9s |
| Covariance | diag |

### Pitman-Yor Process (Pyro/GPU)
| Metric | Value |
|--------|-------|
| K inferred | 3 |
| K error | 27 |
| NMI | 0.2856 |
| ARI | 0.0970 |
| Time | 117.4s |
| Discount | 0.25 |
| Concentration | 1.0 |
| Steps | 1000 |

### PYP Training Loss
```
Step 200/1000:  ELBO loss = 14636064.48
Step 400/1000:  ELBO loss = 14647618.54
Step 600/1000:  ELBO loss = 14678498.71
Step 800/1000:  ELBO loss = 14623484.57
Step 1000/1000: ELBO loss = 14448615.99
```

---

## Summary Comparison

| Metric | DP | PYP | Difference |
|--------|-----|-----|------------|
| K inferred | 30 | 3 | DP +27 |
| NMI | 1.0000 | 0.2856 | DP +71.4% |
| ARI | 1.0000 | 0.0970 | DP +90.3% |
| Time (s) | 119.9 | 117.4 | ~Equal |

---

## Conclusion

```
❌ PYP shows minimal improvement (-71.4%)
   Reports NOT validated for this configuration
```

### Analysis

1. **DP perfectly recovered all 30 clusters** - sklearn's implementation is robust
2. **PYP collapsed to 3 clusters** - variational inference stuck in local optimum
3. **Synthetic data was "too easy"** - well-separated clusters favor any method

### Likely Reasons for PYP Underperformance

1. **Hyperparameter sensitivity** - discount=0.25 may not be optimal
2. **Initialization** - random init led to poor local minimum
3. **Learning rate** - 0.01 may be too high/low
4. **Iterations** - 1000 steps may be insufficient for convergence

### Next Steps

- [ ] Tune PYP hyperparameters (discount, concentration, learning rate)
- [ ] Increase cluster overlap (reduce separation) to make problem harder
- [ ] Try different initializations (K-means, spectral)
- [ ] Test on real trading strategy data where structure is messier

---

## Recommendation for HIMARI

**Use sklearn's Dirichlet Process** for initial implementation:
- Proven performance (NMI=1.0 on test data)
- Fast (~2 min for 100K samples)
- No GPU required
- Simple API

PYP may offer benefits on real data with more complex structure, but requires significant tuning effort.
