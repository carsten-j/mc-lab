import numpy as np

from mc_lab.advanced_rejection_sampling import (
    AdaptiveRejectionSampler,
    SqueezedRejectionSampler,
    TransformedRejectionSampler,
)


# Example usage and comparison
def test_compare_advanced_samplers():
    """Compare different advanced rejection sampling methods."""

    # Target: Standard normal distribution
    def normal_pdf(x):
        return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)

    def log_normal_pdf(x):
        return -0.5 * x * x - 0.5 * np.log(2 * np.pi)

    n_samples = 10_000

    print("Comparing Advanced Rejection Sampling Methods")
    print("=" * 50)

    # 1. Adaptive Rejection Sampling
    print("\n1. Adaptive Rejection Sampling:")
    ars = AdaptiveRejectionSampler(log_normal_pdf, (-5, 5))
    ars_samples = ars.sample(n_samples)
    ars_rate = ars.get_acceptance_rate()
    print(f"   Acceptance rate: {ars_rate:.3f}")
    print(f"   Sample mean: {np.mean(ars_samples):.3f} (should be ~0)")
    print(f"   Sample std: {np.std(ars_samples):.3f} (should be ~1)")

    # 2. Squeezed Rejection Sampling
    print("\n2. Squeezed Rejection Sampling:")
    srs = SqueezedRejectionSampler(normal_pdf, log_normal_pdf, (-5, 5))
    srs_samples = srs.sample(n_samples)
    srs_stats = srs.get_efficiency_stats()
    print(f"   PDF evaluations: {srs_stats['pdf_evaluations']}")
    print(f"   PDF evaluation rate: {srs_stats['pdf_evaluation_rate']:.3f}")
    print(f"   Squeeze acceptance rate: {srs_stats['squeeze_acceptance_rate']:.3f}")
    print(f"   Sample mean: {np.mean(srs_samples):.3f}")
    print(f"   Sample std: {np.std(srs_samples):.3f}")

    # 3. Transformed Rejection Sampling
    print("\n3. Transformed Rejection Sampling (Ratio-of-Uniforms):")
    trs = TransformedRejectionSampler(normal_pdf, mode=0.0)
    trs_samples = trs.sample(n_samples)
    trs_rate = trs.get_acceptance_rate()
    print(f"   Acceptance rate: {trs_rate:.3f}")
    print(f"   Sample mean: {np.mean(trs_samples):.3f}")
    print(f"   Sample std: {np.std(trs_samples):.3f}")

    # return {
    #     'ars': ars_samples,
    #     'srs': srs_samples,
    #     'trs': trs_samples
    # }
