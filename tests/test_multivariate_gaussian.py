import numpy as np

from mc_lab.multivariate_gaussian import sample_multivariate_gaussian


def test_multivariate_gaussian_mean_cov_cholesky():
    rng_seed = 123
    d = 3
    mu = np.array([1.0, -2.0, 0.5])
    A = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 1.5, 0.0],
            [-0.3, 0.2, 0.8],
        ]
    )
    Sigma = A @ A.T

    n = 200_000
    X = sample_multivariate_gaussian(mu, Sigma, n, random_state=rng_seed)
    assert X.shape == (n, d)

    # Sample stats
    m_hat = X.mean(axis=0)
    S_hat = np.cov(X, rowvar=False, bias=False)

    assert np.allclose(m_hat, mu, atol=0.02)
    assert np.allclose(S_hat, Sigma, atol=0.05)


def test_multivariate_gaussian_eigh_fallback():
    # Construct a PSD covariance with a zero eigenvalue
    Q = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank-1
    mu = np.array([0.0, 0.0])
    n = 100_000
    X = sample_multivariate_gaussian(mu, Q, n, random_state=7, method="eigh")
    assert X.shape == (n, 2)
    # Covariance should be close to Q (up to sampling noise)
    S_hat = np.cov(X, rowvar=False, bias=False)
    assert np.allclose(S_hat, Q, atol=0.05)
