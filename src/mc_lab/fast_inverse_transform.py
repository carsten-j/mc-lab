import numpy as np
from scipy.fft import fft, ifft


class ChebyshevSampler:
    """
    A class for inverse transform sampling using Chebyshev polynomial technology.

    This implementation provides sophisticated sampling from arbitrary probability
    density functions using Chebyshev approximations and adaptive domain refinement.

        References:
        - Sheehan Olver and Alex Townsend, "Fast inverse transform sampling in one and two dimensions",
            arXiv:1307.1223 (2013). DOI: https://doi.org/10.48550/arXiv.1307.1223
            https://arxiv.org/abs/1307.1223

        Origin:
        - This implementation is a Python port of the original MATLAB code from
            https://github.com/dlfivefifty/InverseTransformSampling

            Note on coding style:
            - As a direct port focused on algorithmic fidelity and performance, some
                structures and naming may not strictly follow typical Pythonic conventions.
    """

    def __init__(self, pdf, domain, tolerance=100):
        """
        Initialize the Chebyshev sampler.

        Parameters:
        -----------
        pdf : callable
            Probability density function to sample from
        domain : list or tuple
            Domain [a, b] over which the PDF is defined
        tolerance : float, optional
            Tolerance factor for convergence (default: 100)
        """
        self.pdf = pdf
        self.domain = np.array(domain)
        self.tolerance = tolerance
        self.eps = np.finfo(float).eps

        # Cache for computed coefficients
        self._Y_cache = None
        self._refined_domain = None

    def sample(self, N):
        """
        Generate N random samples from the PDF.

        Parameters:
        -----------
        N : int
            Number of samples to generate

        Returns:
        --------
        numpy.ndarray
            Array of N samples from the distribution
        """
        return self._sample_withoutOOP(N)

    def _sample_withoutOOP(self, N):
        """
        Internal method for univariate sampling using Chebyshev technology.
        """
        # Initial domain mapping and normalization
        Y, refined_domain = self._prepare_distribution()

        # Generate samples using the refined distribution
        def map_func_refined(t):
            return (refined_domain[1] - refined_domain[0]) * (
                t + 1
            ) / 2 + refined_domain[0]

        samples_normalized = self._generate_random_samples(Y, N)
        samples = map_func_refined(samples_normalized)

        return samples

    def _prepare_distribution(self):
        """
        Prepare and normalize the distribution, including domain refinement.
        """
        if self._Y_cache is not None and self._refined_domain is not None:
            return self._Y_cache, self._refined_domain

        # Initial domain mapping
        def map_func(t):
            return (self.domain[1] - self.domain[0]) * (t + 1) / 2 + self.domain[0]

        def f(x):
            return self.pdf(map_func(x))

        # Construct Chebyshev approximation
        Y = self._simple_constructor(f)
        out = self._sum_unit_interval(Y)
        Y = Y / out

        # Refine domain based on CDF analysis
        refined_domain = self._refine_domain(Y, map_func)

        # Reconstruct with refined domain
        def map_func_refined(t):
            return (refined_domain[1] - refined_domain[0]) * (
                t + 1
            ) / 2 + refined_domain[0]

        def f_refined(x):
            return self.pdf(map_func_refined(x))

        Y = self._simple_constructor(f_refined)
        out = self._sum_unit_interval(Y)
        Y = Y / out

        # Cache results
        self._Y_cache = Y
        self._refined_domain = refined_domain

        return Y, refined_domain

    def _refine_domain(self, Y, map_func):
        """
        Refine the domain based on CDF analysis to improve sampling accuracy.
        """
        # Compute CDF
        cout = self._simple_cumsum(Y)
        cdf = cout

        # Evaluate CDF at Chebyshev points
        v = self._simple_chebpolyval(cdf)
        tol = self.tolerance * self.eps

        # Find effective domain bounds
        idx1 = np.where(v > tol)[0]
        idx2 = np.where(v < 1 - tol)[0]

        if len(idx1) > 0:
            idx1 = idx1[0]  # first occurrence
        else:
            idx1 = 0

        if len(idx2) > 0:
            idx2 = idx2[-1]  # last occurrence
        else:
            idx2 = len(v) - 1

        # Compute refined domain
        k = len(v) - 1
        x = np.sin(np.pi * np.arange(-k, k + 1, 2) / (2 * k))
        refined_domain = map_func(np.array([x[idx1], x[idx2]]))

        return refined_domain

    def _generate_random_samples(self, Y, N):
        """
        Generate random samples using inverse transform sampling with bisection.
        """
        # Compute CDF
        cout = self._simple_cumsum(Y)
        cdf = cout

        # Generate uniform random samples
        r = np.random.rand(N)

        # Bisection method for inverse CDF
        a = -np.ones(N)
        b = np.ones(N)

        while np.linalg.norm(b - a, ord=np.inf) > 1e-14:
            mid = (a + b) / 2
            vals = self._clenshaw_evaluate(cdf, mid)

            I1 = (vals - r) <= -1e-14
            I2 = (vals - r) >= 1e-14

            a = np.where(I1, mid, np.where(I2, a, mid))
            b = np.where(I1, b, np.where(I2, mid, mid))

        return (a + b) / 2

    def _simple_cumsum(self, Y):
        """
        Compute cumulative sum using Chebyshev coefficients.
        """
        Y = Y[::-1]  # Reverse array
        n = len(Y)
        c = np.concatenate([[0, 0], Y])
        cout = np.zeros(n + 1)

        # Compute coefficients
        if n > 1:
            cout[: n - 1] = (c[2 : n + 1] - c[: n - 1]) / (2 * np.arange(n, 1, -1))

        cout[n - 1] = c[-1] - (c[-3] / 2 if len(c) >= 3 else 0)

        # Compute C_0
        v = np.ones(n)
        v[n - 2 :: -2] = -1
        cout[n] = np.dot(v, cout[:n])

        return cout

    def _simple_constructor(self, f):
        """
        Adaptive Chebyshev constructor with convergence checking.
        """
        for k_exp in range(3, 19):
            k = 2**k_exp
            x = np.sin(np.pi * np.arange(-k, k + 1, 2) / (2 * k))

            vals = f(x)

            # Laurent fold and FFT
            Y = np.concatenate([vals[::-1], vals[1:-1]])
            Y = fft(Y) / (2 * len(x) - 2)
            Y = np.real(Y)
            Y = Y[:k]

            # Convergence check
            threshold = 10 * np.log2(k) * self.eps
            significant_coeffs = np.where(np.abs(Y) > threshold)[0]

            if len(significant_coeffs) > 0 and significant_coeffs[-1] < k - 3:
                Y = Y[: significant_coeffs[-1] + 1]
                if len(Y) > 2:
                    Y[1:-1] = 2 * Y[1:-1]
                return Y

        raise RuntimeError("Chebyshev constructor failed to converge")

    def _clenshaw_evaluate(self, c, x):
        """
        Evaluate Chebyshev series using Clenshaw's recurrence algorithm.
        """
        bk1 = np.zeros_like(x)
        bk2 = np.zeros_like(x)
        x_scaled = 2 * x

        for k in range(len(c) - 1):
            bk = c[k] + x_scaled * bk1 - bk2
            bk2 = bk1
            bk1 = bk

        return c[-1] + 0.5 * x_scaled * bk1 - bk2

    def _sum_unit_interval(self, c):
        """
        Compute integral over unit interval using Chebyshev coefficients.
        """
        n = len(c)
        if n == 1:
            return c[0] * 2

        c_copy = c.copy()
        c_copy[1::2] = 0  # Zero out odd-indexed coefficients

        # Integration weights
        weights = np.zeros(n)
        weights[0] = 2
        if n > 2:
            weights[2:] = 2 / (1 - np.arange(2, n) ** 2)

        return np.dot(weights, c_copy)

    def _simple_chebpolyval(self, c):
        """
        Convert Chebyshev coefficients to values at Chebyshev points.
        """
        c = np.array(c).flatten()
        lc = len(c)

        if lc == 1:
            return c

        # Coefficient modification
        c_mod = c.copy()
        if lc > 2:
            c_mod[1 : lc - 1] = 0.5 * c_mod[1 : lc - 1]

        # Symmetric extension for IFFT
        if lc > 2:
            v = np.concatenate([c_mod[::-1], c_mod[1 : lc - 1]])
        else:
            v = c_mod[::-1]

        # IFFT computation
        if np.all(np.isreal(c)):
            v = np.real(ifft(v))
        elif np.all(np.isreal(1j * c)):
            v = 1j * np.real(ifft(np.imag(v)))
        else:
            v = ifft(v)

        # Final transformation
        if lc > 2:
            result = np.zeros(lc)
            result[0] = 2 * v[0]
            if lc > 2:
                ii = np.arange(1, lc - 1)
                result[ii] = v[ii] + v[2 * lc - 2 - ii]
            result[lc - 1] = 2 * v[lc - 1] if lc > 1 else 0
            v = (lc - 1) * result
        else:
            v = (lc - 1) * np.array([2 * v[0], 2 * v[1] if len(v) > 1 else 0])

        return v[::-1]

    def reset_cache(self):
        """
        Reset internal cache. Call this if you want to recompute the distribution
        with different parameters.
        """
        self._Y_cache = None
        self._refined_domain = None
