#!/usr/bin/env python3
"""
Performance profiling script for StereographicProjectionSampler1D
"""

import cProfile
import os
import pstats

# Import the SPS implementation
import sys
import time
from io import StringIO

import numpy as np

sys.path.append(os.path.dirname(__file__))
from SPS_bimodal import BimodalDistribution, StereographicProjectionSampler1D


def profile_sps_methods():
    """Profile individual methods of SPS sampler"""

    # Setup
    np.random.seed(42)
    sampler = StereographicProjectionSampler1D(R=1.5, h=0.6)
    bimodal = BimodalDistribution(mu1=-4, sigma1=0.8, mu2=3, sigma2=1.2, weight=0.4)

    # Test data
    n_trials = 50000
    x_values = np.random.normal(0, 2, n_trials)  # Representative x values

    print("=== PROFILING INDIVIDUAL METHODS ===")

    # Profile SP_inverse
    start = time.time()
    for x in x_values[:10000]:  # Smaller sample for method profiling
        sampler.SP_inverse(x)
    sp_inverse_time = time.time() - start
    print(
        f"SP_inverse: {sp_inverse_time:.4f}s for 10K calls = {sp_inverse_time / 10000 * 1e6:.2f} μs/call"
    )

    # Profile SP
    z_values = [sampler.SP_inverse(x) for x in x_values[:1000]]
    start = time.time()
    for z in z_values:
        sampler.SP(z)
    sp_time = time.time() - start
    print(f"SP: {sp_time:.4f}s for 1K calls = {sp_time / 1000 * 1e6:.2f} μs/call")

    # Profile propose_on_sphere
    start = time.time()
    for z in z_values:
        sampler.propose_on_sphere(z)
    propose_time = time.time() - start
    print(
        f"propose_on_sphere: {propose_time:.4f}s for 1K calls = {propose_time / 1000 * 1e6:.2f} μs/call"
    )

    # Profile log_density evaluations
    start = time.time()
    for x in x_values[:10000]:
        bimodal.log_density(x)
    log_density_time = time.time() - start
    print(
        f"log_density: {log_density_time:.4f}s for 10K calls = {log_density_time / 10000 * 1e6:.2f} μs/call"
    )

    # Profile single step
    x_current = 0.0
    start = time.time()
    for _ in range(1000):
        x_current, _ = sampler.step(x_current, bimodal.log_density)
    step_time = time.time() - start
    print(f"step: {step_time:.4f}s for 1K calls = {step_time / 1000 * 1e6:.2f} μs/call")


def profile_full_sampling():
    """Profile full sampling with cProfile"""

    print("\n=== PROFILING FULL SAMPLING ===")

    sampler = StereographicProjectionSampler1D(R=1.5, h=0.6)
    bimodal = BimodalDistribution(mu1=-4, sigma1=0.8, mu2=3, sigma2=1.2, weight=0.4)

    # Setup profiler
    pr = cProfile.Profile()

    # Profile sampling
    pr.enable()
    samples, accept_rate = sampler.sample(0.0, bimodal.log_density, 10000, burn_in=1000)
    pr.disable()

    # Print results
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats()

    print("Top function calls by cumulative time:")
    lines = s.getvalue().split("\n")
    for i, line in enumerate(lines):
        if "cumulative" in line:
            # Print header and next 15 lines
            for j in range(min(15, len(lines) - i)):
                print(lines[i + j])
            break

    print(f"\nSamples generated: {len(samples)}")
    print(f"Acceptance rate: {accept_rate:.2%}")


if __name__ == "__main__":
    profile_sps_methods()
    profile_full_sampling()
