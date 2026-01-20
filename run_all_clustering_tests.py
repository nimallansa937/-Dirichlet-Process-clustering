"""
Clustering Benchmark - Run on Vast.AI
Tests: sklearn DP (tuned), sklearn DP (default), HDBSCAN, Sparse Finite Mixture

Command: python run_all_clustering_tests.py
"""

import numpy as np
import time
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import warnings
warnings.filterwarnings('ignore')

# Try to import hdbscan
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("Warning: hdbscan not installed. Run: pip install hdbscan")

RANDOM_STATE = 42

def generate_synthetic_data(n_samples=50000, n_features=60, n_clusters=30,
                            separation=1.0, zipf_exp=2.0, elliptical=True):
    """Generate power-law distributed clusters with optional elliptical shapes"""
    np.random.seed(RANDOM_STATE)

    # Power-law cluster sizes
    weights = np.array([1.0 / (i ** zipf_exp) for i in range(1, n_clusters + 1)])
    weights /= weights.sum()
    cluster_sizes = np.round(weights * n_samples).astype(int)
    cluster_sizes[-1] = n_samples - cluster_sizes[:-1].sum()  # Fix rounding

    X_list = []
    y_list = []

    for k in range(n_clusters):
        n_k = cluster_sizes[k]
        if n_k <= 0:
            continue

        # Random cluster center
        center = np.random.randn(n_features) * separation * 5

        if elliptical:
            # Random covariance matrix
            A = np.random.randn(n_features, n_features) * 0.3
            cov = A @ A.T + np.eye(n_features) * 0.1
            X_k = np.random.multivariate_normal(center, cov, size=n_k)
        else:
            # Spherical clusters
            X_k = np.random.randn(n_k, n_features) + center

        X_list.append(X_k)
        y_list.append(np.full(n_k, k))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Shuffle
    perm = np.random.permutation(len(X))
    return X[perm], y[perm]

def test_sklearn_dp_tuned(X, y_true):
    """sklearn DP with TUNED sparse prior (α=0.05)"""
    print("  Running sklearn DP (tuned α=0.05)...")
    start = time.time()

    model = BayesianGaussianMixture(
        n_components=100,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=0.05,  # TUNED: sparse prior
        covariance_type='diag',           # Stable in high-d
        init_params='k-means++',
        n_init=5,
        max_iter=500,
        random_state=RANDOM_STATE
    )
    model.fit(X)

    labels = model.predict(X)
    effective_k = np.sum(model.weights_ > 1e-3)
    nmi = normalized_mutual_info_score(y_true, labels)
    ari = adjusted_rand_score(y_true, labels)

    return {
        'method': 'sklearn_DP_tuned',
        'K': int(effective_k),
        'NMI': nmi,
        'ARI': ari,
        'time_sec': time.time() - start
    }

def test_sklearn_dp_default(X, y_true):
    """sklearn DP with DEFAULT prior (α=1.0) - expect over-segmentation"""
    print("  Running sklearn DP (default α=1.0)...")
    start = time.time()

    model = BayesianGaussianMixture(
        n_components=100,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=1.0,  # DEFAULT: causes over-segmentation
        covariance_type='diag',
        init_params='k-means++',
        n_init=5,
        max_iter=500,
        random_state=RANDOM_STATE
    )
    model.fit(X)

    labels = model.predict(X)
    effective_k = np.sum(model.weights_ > 1e-3)
    nmi = normalized_mutual_info_score(y_true, labels)
    ari = adjusted_rand_score(y_true, labels)

    return {
        'method': 'sklearn_DP_default',
        'K': int(effective_k),
        'NMI': nmi,
        'ARI': ari,
        'time_sec': time.time() - start
    }

def test_hdbscan(X, y_true, k_expected=30):
    """HDBSCAN density-based clustering"""
    if not HAS_HDBSCAN:
        return {'method': 'HDBSCAN', 'K': 0, 'NMI': 0, 'ARI': 0, 'time_sec': 0, 'error': 'not installed'}

    print("  Running HDBSCAN...")
    start = time.time()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(len(X) / k_expected / 2),
        min_samples=10,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(X)

    # Filter noise for metrics
    mask = labels != -1
    if mask.sum() > 0:
        nmi = normalized_mutual_info_score(y_true[mask], labels[mask])
        ari = adjusted_rand_score(y_true[mask], labels[mask])
    else:
        nmi, ari = 0, 0

    effective_k = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = 100 * (labels == -1).sum() / len(labels)

    return {
        'method': 'HDBSCAN',
        'K': effective_k,
        'NMI': nmi,
        'ARI': ari,
        'noise_pct': noise_pct,
        'time_sec': time.time() - start
    }

def test_sparse_finite_mixture(X, y_true):
    """Sparse Finite Mixture with BIC model selection"""
    print("  Running Sparse Finite Mixture (BIC selection)...")
    start = time.time()

    best_bic = np.inf
    best_k = None
    best_labels = None

    k_range = [10, 15, 20, 25, 30, 35, 40, 45, 50]

    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='diag',
            n_init=3,
            max_iter=300,
            random_state=RANDOM_STATE
        )
        gmm.fit(X)
        bic = gmm.bic(X)

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_labels = gmm.predict(X)

        print(f"    K={k:2d}: BIC={bic:,.0f}")

    nmi = normalized_mutual_info_score(y_true, best_labels)
    ari = adjusted_rand_score(y_true, best_labels)

    return {
        'method': 'SFM_BIC',
        'K': best_k,
        'NMI': nmi,
        'ARI': ari,
        'BIC': best_bic,
        'time_sec': time.time() - start
    }

def main():
    print("="*70)
    print("BAYESIAN CLUSTERING BENCHMARK")
    print("="*70)

    # Generate data
    print("\nGenerating synthetic data...")
    X_raw, y_true = generate_synthetic_data(
        n_samples=50000,
        n_features=60,
        n_clusters=30,
        separation=1.0,      # Moderate overlap
        zipf_exp=2.0,        # Power-law distribution
        elliptical=True      # Non-spherical clusters
    )

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    true_k = len(np.unique(y_true))
    print(f"Data shape: {X.shape}")
    print(f"True K: {true_k}")
    print(f"Cluster size range: {np.bincount(y_true).min()} - {np.bincount(y_true).max()}")

    # Run tests
    print("\n" + "-"*70)
    print("RUNNING TESTS")
    print("-"*70)

    results = []

    results.append(test_sklearn_dp_tuned(X, y_true))
    results.append(test_sklearn_dp_default(X, y_true))
    results.append(test_hdbscan(X, y_true, k_expected=30))
    results.append(test_sparse_finite_mixture(X, y_true))

    # Print results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Method':<20} {'K':>6} {'K_true':>8} {'NMI':>8} {'ARI':>8} {'Time(s)':>10}")
    print("-"*70)
    for r in results:
        print(f"{r['method']:<20} {r['K']:>6} {true_k:>8} {r['NMI']:>8.4f} {r['ARI']:>8.4f} {r['time_sec']:>10.1f}")
    print("="*70)

    # Validation
    print("\nVALIDATION (K_true=30):")
    for r in results:
        k_ok = 20 <= r['K'] <= 45
        nmi_ok = r['NMI'] >= 0.80
        status = "PASS" if (k_ok and nmi_ok) else "FAIL"
        k_status = "OK" if k_ok else f"FAIL ({r['K']})"
        nmi_status = "OK" if nmi_ok else f"FAIL ({r['NMI']:.2f})"
        print(f"  {r['method']:<20}: K={k_status:<12} NMI={nmi_status:<12} -> {status}")

    # Save results
    print("\nSaving results to clustering_results.txt...")
    with open("clustering_results.txt", "w") as f:
        f.write("CLUSTERING BENCHMARK RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Data: n={X.shape[0]}, d={X.shape[1]}, K_true={true_k}\n\n")
        for r in results:
            f.write(f"{r['method']}:\n")
            f.write(f"  K inferred: {r['K']}\n")
            f.write(f"  NMI: {r['NMI']:.4f}\n")
            f.write(f"  ARI: {r['ARI']:.4f}\n")
            f.write(f"  Time: {r['time_sec']:.1f}s\n\n")

    print("\nDone!")

if __name__ == "__main__":
    main()
