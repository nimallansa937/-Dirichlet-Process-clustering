# Pitman-Yor vs Dirichlet Process Clustering Validation

Empirical validation of claims that Pitman-Yor Process (PYP) outperforms standard Dirichlet Process (DP) mixture models on power-law distributed data.

## Claims Being Tested

| Claim | Testable Prediction |
|-------|---------------------|
| PYP beats DP by 17-22% on power-law data | NMI(PYP) > NMI(DP) by ~0.15-0.20 |
| DP underestimates cluster count | K_inferred(DP) < K_true; K_inferred(PYP) ≈ K_true |
| DP "collapses" niche clusters | Small clusters merged into large ones under DP |
| PYP sustains discovery as n grows | New cluster probability doesn't decay to zero |

## Quick Start

### On Vast.AI with NVIDIA B200

```bash
# Clone repository
git clone https://github.com/nimallansa937/-Dirichlet-Process-clustering.git
cd -Dirichlet-Process-clustering

# Install dependencies
pip install -r requirements.txt

# Run quick test
python validate_pyp_vs_dp.py --n_samples 10000 --quick

# Run full validation
python validate_pyp_vs_dp.py --n_samples 50000 --n_features 60 --n_clusters 30

# Run full parameter sweep (recommended)
python validate_pyp_vs_dp.py --run_sweep --output results.csv
```

### Expected Output

```
================================================================
PYP vs DP CLUSTERING VALIDATION
================================================================
PyTorch: 2.x.x
Pyro:    1.8.x
Device:  NVIDIA B200
GPU Memory: 192.0 GB
================================================================

EXPERIMENT: n=50000, d=60, K=30, zipf=1.5
...

SUMMARY
True K: 30
DP  inferred K: 18 (error: 12)
PYP inferred K: 27 (error: 3)

NMI: DP=0.7234, PYP=0.8567
NMI improvement: +18.4%
```

## Method Comparison

### Dirichlet Process (Baseline)
- Stick-breaking: β_k ~ Beta(1, α)
- Used via sklearn's BayesianGaussianMixture
- CPU-only, variational inference

### Pitman-Yor Process (Test)
- Stick-breaking: β_k ~ Beta(1-d, α + k·d)
- Discount parameter d ∈ [0, 1) modulates "rich-get-richer"
- GPU-accelerated via Pyro
- Better models power-law tails

## Parameter Sweep

The sweep tests across:
- Zipf exponents: [1.2, 1.5, 2.0, 2.5] (power-law steepness)
- Sample sizes: [10K, 50K, 100K]
- Multiple random seeds

## Interpretation Guide

| Observation | Conclusion |
|-------------|------------|
| NMI(PYP) > NMI(DP) by 10%+ | Reports validated - use PYP |
| NMI(PYP) ≈ NMI(DP) (within 5%) | Reports don't generalize - stick with DP |
| K(DP) << K_true, K(PYP) ≈ K_true | Reports validated - DP collapses clusters |
| PYP >5× slower than DP | Weigh computational cost vs quality |

## GPU Requirements

- Minimum: 8GB VRAM (10K samples)
- Recommended: 24GB+ VRAM (100K+ samples)
- Optimal: NVIDIA B200 192GB (500K+ samples)

## License

MIT
