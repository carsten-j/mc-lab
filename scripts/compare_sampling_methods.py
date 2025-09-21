import time

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import gamma, norm

from mc_lab.gibbs_sampler_2d import GibbsSampler2D
from mc_lab.independent_metropolis_hastings import IndependentMetropolisHastingsSampler
from mc_lab.MALA_auto_grad import MALAAutoGradSampler
from mc_lab.metropolis_hastings import MetropolisHastingsSampler


def target_density(x, y):
    """Target probability density (unnormalized): x^2 * exp(-x*y^2 - y^2 + 2*y - 4*x)"""
    return x**2 * np.exp(-x * y**2 - y**2 + 2 * y - 4 * x)


def log_target_density(x, y):
    """Log of target probability density"""
    if x <= 0:  # x must be positive for x^2 term
        return -np.inf
    return 2 * np.log(x) - x * y**2 - y**2 + 2 * y - 4 * x


def log_target_2d(xy):
    """2D log target for Metropolis-Hastings sampler"""
    x, y = xy[0], xy[1]
    return log_target_density(x, y)


def grad_log_target_2d(xy):
    """Gradient of log target density for MALA sampler"""
    x, y = xy[0], xy[1]
    if x <= 0:
        return np.array([-np.inf, -np.inf])

    # Partial derivatives of log_target_density
    grad_x = 2.0 / x - y**2 - 4
    grad_y = -2 * x * y - 2 * y + 2

    return np.array([grad_x, grad_y])


def sample_x_given_y(y):
    """Sample x|y ~ Gamma(3, scale=1/(y^2 + 4))"""
    scale = 1.0 / (y**2 + 4)
    return gamma.rvs(3, scale=scale)


def sample_y_given_x(x):
    """Sample y|x ~ Normal(1/(1+x), scale=1/sqrt(2*(x+1)))"""
    if x <= -1:  # Avoid numerical issues
        x = -0.99
    mean = 1.0 / (1 + x)
    scale = 1.0 / np.sqrt(2 * (x + 1))
    return norm.rvs(mean, scale)


# ============================================================================
# Comprehensive MCMC Diagnostics Using ArviZ
# ============================================================================


def comprehensive_diagnostics(idata, var_names=None):
    """Calculate all major MCMC diagnostics using ArviZ.

    Parameters
    ----------
    idata : arviz.InferenceData
        ArviZ InferenceData object containing MCMC samples
    var_names : list of str, optional
        Names of variables to include in diagnostics

    Returns
    -------
    dict
        Dictionary containing diagnostic results and interpretations
    """
    diagnostics = {}

    # Effective Sample Size (bulk and tail)
    diagnostics["ess_bulk"] = az.ess(idata, var_names=var_names, method="bulk")
    diagnostics["ess_tail"] = az.ess(idata, var_names=var_names, method="tail")

    # R-hat convergence diagnostic (rank-normalized)
    diagnostics["rhat"] = az.rhat(idata, var_names=var_names, method="rank")

    # Monte Carlo Standard Error
    diagnostics["mcse_mean"] = az.mcse(idata, var_names=var_names, method="mean")
    diagnostics["mcse_sd"] = az.mcse(idata, var_names=var_names, method="sd")

    # Autocorrelation function
    posterior_samples = az.extract(idata, var_names=var_names)
    diagnostics["autocorr"] = {
        var: az.autocorr(posterior_samples[var].values)
        for var in posterior_samples.data_vars
    }

    # Comprehensive summary
    diagnostics["summary"] = az.summary(
        idata, var_names=var_names, stat_focus="convergence"
    )

    return diagnostics


def interpret_diagnostics(diagnostics):
    """Provide interpretation of diagnostic results.

    Parameters
    ----------
    diagnostics : dict
        Output from comprehensive_diagnostics()

    Returns
    -------
    dict
        Interpretation of diagnostic results with status indicators
    """
    interpretation = {}

    # R-hat interpretation
    rhat_vals = diagnostics["rhat"].to_array().values.flatten()
    max_rhat = np.max(rhat_vals)
    interpretation["convergence"] = {
        "max_rhat": max_rhat,
        "status": "excellent"
        if max_rhat < 1.01
        else "good"
        if max_rhat < 1.05
        else "poor"
        if max_rhat < 1.1
        else "very_poor",
    }

    # ESS interpretation
    ess_bulk_vals = diagnostics["ess_bulk"].to_array().values.flatten()
    min_ess_bulk = np.min(ess_bulk_vals)
    interpretation["efficiency"] = {
        "min_ess_bulk": min_ess_bulk,
        "status": "excellent"
        if min_ess_bulk > 1000
        else "good"
        if min_ess_bulk > 400
        else "adequate"
        if min_ess_bulk > 100
        else "poor",
    }

    return interpretation


def sampler_efficiency_comparison(idata_dict):
    """Compare sampling efficiency across multiple samplers.

    Parameters
    ----------
    idata_dict : dict
        Dictionary of {sampler_name: InferenceData}

    Returns
    -------
    dict
        Efficiency metrics for each sampler
    """
    efficiency_metrics = {}

    for sampler_name, idata in idata_dict.items():
        # Basic efficiency metrics
        ess_bulk = az.ess(idata, method="bulk")
        ess_tail = az.ess(idata, method="tail")
        rhat = az.rhat(idata)

        n_chains = idata.posterior.sizes["chain"]
        n_draws = idata.posterior.sizes["draw"]
        total_samples = n_chains * n_draws

        # Efficiency ratios
        bulk_efficiency = ess_bulk.to_array().min() / total_samples
        tail_efficiency = ess_tail.to_array().min() / total_samples

        # Convergence assessment
        max_rhat = float(rhat.to_array().max())
        converged = max_rhat < 1.05

        efficiency_metrics[sampler_name] = {
            "total_samples": total_samples,
            "min_ess_bulk": float(ess_bulk.to_array().min()),
            "min_ess_tail": float(ess_tail.to_array().min()),
            "bulk_efficiency": float(bulk_efficiency),
            "tail_efficiency": float(tail_efficiency),
            "max_rhat": max_rhat,
            "converged": converged,
        }

        # Add acceptance rate if available
        if hasattr(idata, "sample_stats") and "accept_rate" in idata.sample_stats:
            efficiency_metrics[sampler_name]["mean_accept_rate"] = float(
                idata.sample_stats["accept_rate"].mean()
            )
        elif hasattr(idata, "sample_stats") and "accepted" in idata.sample_stats:
            # Calculate from accepted/rejected indicators
            efficiency_metrics[sampler_name]["mean_accept_rate"] = float(
                idata.sample_stats["accepted"].mean()
            )

    return efficiency_metrics


def rank_samplers(efficiency_metrics):
    """Rank samplers based on multiple criteria.

    Parameters
    ----------
    efficiency_metrics : dict
        Output from sampler_efficiency_comparison()

    Returns
    -------
    dict
        Rankings for different criteria and overall ranking
    """
    ranking_criteria = {
        "convergence": lambda x: -x["max_rhat"],  # Lower R-hat is better
        "bulk_efficiency": lambda x: x["bulk_efficiency"],  # Higher is better
        "tail_efficiency": lambda x: x["tail_efficiency"],  # Higher is better
    }

    # Add speed criterion if runtime information is available
    if "runtime" in list(efficiency_metrics.values())[0]:
        ranking_criteria["speed"] = lambda x: x["samples_per_second"]

    rankings = {}
    for criterion, score_func in ranking_criteria.items():
        sorted_samplers = sorted(
            efficiency_metrics.items(), key=lambda x: score_func(x[1]), reverse=True
        )
        rankings[criterion] = [name for name, _ in sorted_samplers]

    # Calculate overall ranking (simple average of ranks)
    overall_scores = {}
    for sampler_name in efficiency_metrics:
        rank_sum = 0
        for criterion_rankings in rankings.values():
            rank_sum += criterion_rankings.index(sampler_name)
        overall_scores[sampler_name] = rank_sum / len(rankings)

    overall_ranking = sorted(overall_scores.items(), key=lambda x: x[1])
    rankings["overall"] = [name for name, _ in overall_ranking]

    return rankings


# ============================================================================
# Advanced Visualization Functions for Sampler Comparison
# ============================================================================


def create_comparison_plots(idata_dict, var_names=None):
    """Create comprehensive comparison plots for multiple samplers.

    Parameters
    ----------
    idata_dict : dict
        Dictionary of {sampler_name: InferenceData}
    var_names : list of str, optional
        Variables to include in plots
    """
    n_samplers = len(idata_dict)

    # 1. Trace plot comparison
    for i, (name, idata) in enumerate(idata_dict.items()):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        az.plot_trace(idata, var_names=var_names, axes=axes)
        fig.suptitle(f"{name} - Trace Plot", size=16)
        plt.tight_layout()
        plt.show()

    # 2. Rank plot comparison (more sensitive than trace plots)
    fig, axes = plt.subplots(1, n_samplers, figsize=(4 * n_samplers, 6))
    if n_samplers == 1:
        axes = [axes]

    for i, (name, idata) in enumerate(idata_dict.items()):
        az.plot_rank(idata, var_names=var_names, ax=axes[i])
        axes[i].set_title(f"{name}\nRank Plot")

    plt.tight_layout()
    plt.show()

    # 3. ESS comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bulk ESS
    ess_data = {}
    for name, idata in idata_dict.items():
        ess_bulk = az.ess(idata, var_names=var_names, method="bulk")
        ess_data[name] = ess_bulk.to_array().values.flatten()

    # Box plot of ESS values
    axes[0].boxplot(list(ess_data.values()), tick_labels=list(ess_data.keys()))
    axes[0].set_ylabel("Effective Sample Size (Bulk)")
    axes[0].set_title("Bulk ESS Comparison")
    axes[0].axhline(
        y=400, color="red", linestyle="--", alpha=0.7, label="Recommended minimum"
    )
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=45)

    # Tail ESS
    ess_tail_data = {}
    for name, idata in idata_dict.items():
        ess_tail = az.ess(idata, var_names=var_names, method="tail")
        ess_tail_data[name] = ess_tail.to_array().values.flatten()

    axes[1].boxplot(
        list(ess_tail_data.values()), tick_labels=list(ess_tail_data.keys())
    )
    axes[1].set_ylabel("Effective Sample Size (Tail)")
    axes[1].set_title("Tail ESS Comparison")
    axes[1].axhline(
        y=100, color="red", linestyle="--", alpha=0.7, label="Recommended minimum"
    )
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    # 4. Autocorrelation comparison
    for i, (name, idata) in enumerate(idata_dict.items()):
        fig = plt.figure(figsize=(10, 6))
        az.plot_autocorr(idata, var_names=var_names, max_lag=50)
        fig.suptitle(f"{name} - Autocorrelation", size=16)
        plt.tight_layout()
        plt.show()


def plot_diagnostic_summary(efficiency_metrics):
    """Create summary visualization of efficiency metrics.

    Parameters
    ----------
    efficiency_metrics : dict
        Output from sampler_efficiency_comparison()
    """
    samplers = list(efficiency_metrics.keys())
    metrics = ["bulk_efficiency", "tail_efficiency", "max_rhat"]

    # Add acceptance rate if available
    if "mean_accept_rate" in efficiency_metrics[samplers[0]]:
        metrics.append("mean_accept_rate")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        if i >= len(axes):
            break

        values = [efficiency_metrics[sampler].get(metric, 0) for sampler in samplers]

        bars = axes[i].bar(samplers, values)
        axes[i].set_title(metric.replace("_", " ").title())
        axes[i].tick_params(axis="x", rotation=45)

        # Add reference lines for some metrics
        if metric == "max_rhat":
            axes[i].axhline(
                y=1.01, color="green", linestyle="--", alpha=0.7, label="Excellent"
            )
            axes[i].axhline(
                y=1.05, color="orange", linestyle="--", alpha=0.7, label="Good"
            )
            axes[i].axhline(y=1.1, color="red", linestyle="--", alpha=0.7, label="Poor")
            axes[i].legend()
        elif "efficiency" in metric:
            axes[i].set_ylabel("ESS / Total Samples")
        elif metric == "mean_accept_rate":
            axes[i].set_ylabel("Acceptance Rate")
            axes[i].set_ylim(0, 1)

        # Color bars based on performance
        if metric == "max_rhat":
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val < 1.01:
                    bar.set_color("green")
                elif val < 1.05:
                    bar.set_color("orange")
                else:
                    bar.set_color("red")
        elif "efficiency" in metric:
            for j, (bar, val) in enumerate(zip(bars, values)):
                if val > 0.5:
                    bar.set_color("green")
                elif val > 0.2:
                    bar.set_color("orange")
                else:
                    bar.set_color("red")

    # Hide unused subplots
    for i in range(len(metrics), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_mixing_comparison(idata_dict, param_name="x"):
    """Compare chain mixing across samplers for a specific parameter.

    Parameters
    ----------
    idata_dict : dict
        Dictionary of {sampler_name: InferenceData}
    param_name : str
        Name of parameter to plot
    """
    fig, axes = plt.subplots(len(idata_dict), 1, figsize=(12, 3 * len(idata_dict)))
    if len(idata_dict) == 1:
        axes = [axes]

    for i, (sampler_name, idata) in enumerate(idata_dict.items()):
        if param_name in idata.posterior:
            # Plot each chain separately to visualize mixing
            chains = idata.posterior[param_name]
            n_chains = chains.sizes["chain"]
            for chain_idx in range(n_chains):
                axes[i].plot(
                    chains.isel(chain=chain_idx),
                    alpha=0.7,
                    label=f"Chain {chain_idx + 1}",
                )

            axes[i].set_title(f"{sampler_name} - {param_name} Chain Mixing")
            axes[i].set_xlabel("Iteration")
            axes[i].set_ylabel(param_name)
            if n_chains > 1:
                axes[i].legend()

    plt.tight_layout()
    plt.show()


def plot_posterior_comparison(idata_dict, var_names=None):
    """Compare posterior distributions across samplers.

    Parameters
    ----------
    idata_dict : dict
        Dictionary of {sampler_name: InferenceData}
    var_names : list of str, optional
        Variables to compare
    """
    if not var_names:
        var_names = list(list(idata_dict.values())[0].posterior.data_vars)

    n_vars = len(var_names)
    n_samplers = len(idata_dict)

    fig, axes = plt.subplots(n_vars, 1, figsize=(10, 3 * n_vars))
    if n_vars == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_samplers))

    for i, var_name in enumerate(var_names):
        for j, (sampler_name, idata) in enumerate(idata_dict.items()):
            if var_name in idata.posterior:
                # Extract samples using a simpler approach
                samples_data = idata.posterior[var_name].values.flatten()
                axes[i].hist(
                    samples_data,
                    bins=50,
                    alpha=0.6,
                    color=colors[j],
                    label=sampler_name,
                    density=True,
                )

        axes[i].set_title(f"Posterior Distribution: {var_name}")
        axes[i].set_xlabel(var_name)
        axes[i].set_ylabel("Density")
        axes[i].legend()

    plt.tight_layout()
    plt.show()


# ============================================================================
# Standardized Fair Comparison Framework
# ============================================================================


def fair_sampler_comparison(
    samplers_dict,
    target_func,
    initial_value,
    n_samples=2000,
    n_chains=4,
    var_names=None,
):
    """
    Conduct fair comparison between MCMC samplers following ArviZ best practices.

    Parameters
    ----------
    samplers_dict : dict
        Dictionary of {name: sampler_instance}
    target_func : callable
        Log target distribution function
    initial_value : array
        Starting values for all samplers
    n_samples : int
        Number of samples per chain
    n_chains : int
        Number of chains to run
    var_names : list, optional
        Parameter names for ArviZ

    Returns
    -------
    tuple
        (results_dict, efficiency_metrics, runtimes)
    """
    results = {}
    runtimes = {}

    # Set same random seed base for reproducibility
    base_seed = 42

    print("=== Running MCMC Sampler Comparison ===")

    for sampler_name, sampler in samplers_dict.items():
        print(f"Running {sampler_name}...")
        np.random.seed(base_seed)

        start_time = time.time()

        # Run sampler using the standardized interface
        idata = sampler.sample(
            n_samples=n_samples,
            n_chains=n_chains,
            burn_in=max(500, n_samples // 4),  # Adaptive burn-in
            initial_states=initial_value,
            progressbar=False,
        )

        end_time = time.time()
        runtimes[sampler_name] = end_time - start_time
        results[sampler_name] = idata

        print(f"  Completed in {runtimes[sampler_name]:.2f} seconds")

    # Calculate comprehensive diagnostics
    efficiency_metrics = sampler_efficiency_comparison(results)

    # Add runtime information
    for sampler_name in efficiency_metrics:
        efficiency_metrics[sampler_name]["runtime"] = runtimes[sampler_name]
        efficiency_metrics[sampler_name]["samples_per_second"] = (
            n_chains * n_samples / runtimes[sampler_name]
        )

    return results, efficiency_metrics, runtimes


def print_comparison_summary(efficiency_metrics, rankings):
    """Print a comprehensive summary of the comparison results.

    Parameters
    ----------
    efficiency_metrics : dict
        Output from sampler_efficiency_comparison()
    rankings : dict
        Output from rank_samplers()
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SAMPLER COMPARISON RESULTS")
    print("=" * 80)

    print(
        f"\n{'Rank':<4} {'Sampler':<20} {'R-hat':<8} {'ESS(bulk)':<10} {'ESS(tail)':<10} {'Accept':<8} {'Time(s)':<8}"
    )
    print("-" * 80)

    for i, sampler_name in enumerate(rankings["overall"]):
        metrics = efficiency_metrics[sampler_name]
        accept_rate = metrics.get("mean_accept_rate", "N/A")
        accept_str = f"{accept_rate:.3f}" if accept_rate != "N/A" else accept_rate

        print(
            f"{i + 1:<4} {sampler_name:<20} {metrics['max_rhat']:<8.4f} "
            f"{metrics['min_ess_bulk']:<10.0f} {metrics['min_ess_tail']:<10.0f} "
            f"{accept_str:<8} {metrics['runtime']:<8.2f}"
        )

    print("\n" + "=" * 80)
    print("CONVERGENCE ASSESSMENT")
    print("=" * 80)

    for sampler_name in rankings["overall"]:
        metrics = efficiency_metrics[sampler_name]
        status = "✓ CONVERGED" if metrics["converged"] else "✗ NOT CONVERGED"
        rhat_status = (
            "Excellent"
            if metrics["max_rhat"] < 1.01
            else "Good"
            if metrics["max_rhat"] < 1.05
            else "Poor"
            if metrics["max_rhat"] < 1.1
            else "Very Poor"
        )

        ess_status = (
            "Excellent"
            if metrics["min_ess_bulk"] > 1000
            else "Good"
            if metrics["min_ess_bulk"] > 400
            else "Adequate"
            if metrics["min_ess_bulk"] > 100
            else "Poor"
        )

        print(f"\n{sampler_name}:")
        print(f"  {status}")
        print(f"  R-hat: {metrics['max_rhat']:.4f} ({rhat_status})")
        print(f"  ESS: {metrics['min_ess_bulk']:.0f} ({ess_status})")
        print(f"  Efficiency: {metrics['bulk_efficiency']:.4f}")

        if "mean_accept_rate" in metrics:
            print(f"  Acceptance Rate: {metrics['mean_accept_rate']:.3f}")


def run_independent_mh():
    """Run Independent Metropolis-Hastings sampler using the class interface"""
    print("Running Independent Metropolis-Hastings...")

    # Use a bivariate normal proposal centered at (1, 1) with moderate spread
    proposal_mean = np.array([1.0, 1.0])
    proposal_cov = np.array([[0.5, 0.1], [0.1, 0.3]])
    proposal_dist = stats.multivariate_normal(proposal_mean, proposal_cov)

    def target_log_pdf(xy):
        return log_target_2d(xy)

    def proposal_sampler():
        return proposal_dist.rvs()

    def proposal_log_pdf(xy):
        return proposal_dist.logpdf(xy)

    # Use the class-based interface for 2D sampling
    sampler = IndependentMetropolisHastingsSampler(
        target_log_pdf=target_log_pdf,
        proposal_sampler=proposal_sampler,
        proposal_log_pdf=proposal_log_pdf,
        var_names=["x", "y"],
    )

    idata = sampler.sample(
        n_samples=10000,
        n_chains=1,
        burn_in=2000,
        initial_states=np.array([1.0, 1.0]),
        progressbar=False,
    )

    # Extract samples
    samples = np.column_stack(
        [idata.posterior["x"].values[0], idata.posterior["y"].values[0]]
    )

    acceptance_rates = sampler.get_acceptance_rates(idata)
    acceptance_rate = acceptance_rates["overall"]
    print(f"Independent MH acceptance rate: {acceptance_rate:.3f}")

    return samples, acceptance_rate


def run_gibbs_sampler():
    """Run Gibbs sampler"""
    print("Running Gibbs sampler...")

    def log_joint(x, y):
        return log_target_density(x, y)

    sampler = GibbsSampler2D(
        sample_x_given_y=sample_x_given_y,
        sample_y_given_x=sample_y_given_x,
        log_target=log_joint,
        var_names=("x", "y"),
    )

    idata = sampler.sample(
        n_samples=10000,
        n_chains=1,
        burn_in=2000,
        initial_states=np.array([1.0, 1.0]),
        progressbar=False,
    )

    # Extract samples
    samples = np.column_stack(
        [idata.posterior["x"].values[0], idata.posterior["y"].values[0]]
    )

    print("Gibbs sampler acceptance rate: 1.000 (always accepts)")

    return samples, 1.0


def run_metropolis_hastings():
    """Run Metropolis-Hastings with Random Walk"""
    print("Running Metropolis-Hastings with Random Walk...")

    sampler = MetropolisHastingsSampler(
        log_target=log_target_2d,
        proposal_scale=np.array([0.2, 0.15]),  # Tuned for reasonable acceptance
        var_names=["x", "y"],
        adaptive_scaling=True,
    )

    idata = sampler.sample(
        n_samples=10000,
        n_chains=1,
        burn_in=2000,
        initial_states=np.array([1.0, 1.0]),
        progressbar=False,
    )

    # Extract samples
    samples = np.column_stack(
        [idata.posterior["x"].values[0], idata.posterior["y"].values[0]]
    )

    acceptance_rates = sampler.get_acceptance_rates(idata)
    acceptance_rate = acceptance_rates["overall"]
    print(f"Metropolis-Hastings acceptance rate: {acceptance_rate:.3f}")

    return samples, acceptance_rate


def run_mala_sampler():
    """Run MALA (Metropolis-adjusted Langevin Algorithm) with automatic gradients"""
    print("Running MALA sampler with automatic gradients...")

    # Import torch for PyTorch-compatible log target
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for MALAAutoGradSampler")

    def log_target_torch(xy):
        """PyTorch-compatible log target function"""
        x, y = xy[0], xy[1]
        # Use torch.where to handle x <= 0 case in a differentiable way
        # Add small epsilon to avoid log(0) and ensure x > 0
        x_safe = torch.where(x > 0, x, torch.tensor(1e-10, dtype=x.dtype))
        log_prob = 2 * torch.log(x_safe) - x_safe * y**2 - y**2 + 2 * y - 4 * x_safe
        # Return very negative value for x <= 0
        return torch.where(x > 0, log_prob, torch.tensor(-1e10, dtype=x.dtype))

    sampler = MALAAutoGradSampler(
        log_target=log_target_torch,
        step_size=0.15,  # Tuned for reasonable acceptance and good mixing
        var_names=["x", "y"],
    )

    idata = sampler.sample(
        n_samples=10000,
        n_chains=1,
        burn_in=2000,  # Longer burn-in for better convergence
        initial_states=np.array([0.5, 0.5]),  # Start closer to the mode
        progressbar=False,
    )

    # Extract samples
    samples = np.column_stack(
        [idata.posterior["x"].values[0], idata.posterior["y"].values[0]]
    )

    acceptance_rates = sampler.get_acceptance_rates(idata)
    acceptance_rate = acceptance_rates["overall"]
    print(f"MALA acceptance rate: {acceptance_rate:.3f}")

    return samples, acceptance_rate


def compare_methods_arviz():
    """Compare all four sampling methods using comprehensive ArviZ diagnostics."""

    # Initialize samplers with optimized parameters
    samplers_dict = initialize_samplers()

    # Run standardized comparison
    results, efficiency_metrics, runtimes = fair_sampler_comparison(
        samplers_dict=samplers_dict,
        target_func=log_target_2d,
        initial_value=np.array([1.0, 1.0]),
        n_samples=2000,
        n_chains=4,
        var_names=["x", "y"],
    )

    # Calculate rankings
    rankings = rank_samplers(efficiency_metrics)

    # Print comprehensive summary
    print_comparison_summary(efficiency_metrics, rankings)

    # Create comprehensive diagnostic plots
    print("\n" + "=" * 80)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("=" * 80)

    # 1. Comprehensive comparison plots
    create_comparison_plots(results, var_names=["x", "y"])

    # 2. Diagnostic summary visualization
    plot_diagnostic_summary(efficiency_metrics)

    # 3. Chain mixing comparison
    plot_mixing_comparison(results, param_name="x")
    plot_mixing_comparison(results, param_name="y")

    # 4. Posterior distribution comparison
    plot_posterior_comparison(results, var_names=["x", "y"])

    return results, efficiency_metrics, rankings


def initialize_samplers():
    """Initialize all MCMC samplers with optimized parameters."""

    # Independent Metropolis-Hastings
    proposal_mean = np.array([1.0, 1.0])
    proposal_cov = np.array([[0.5, 0.1], [0.1, 0.3]])
    proposal_dist = stats.multivariate_normal(proposal_mean, proposal_cov)

    imh_sampler = IndependentMetropolisHastingsSampler(
        target_log_pdf=log_target_2d,
        proposal_sampler=proposal_dist.rvs,
        proposal_log_pdf=proposal_dist.logpdf,
        var_names=["x", "y"],
    )

    # Gibbs Sampler
    gibbs_sampler = GibbsSampler2D(
        sample_x_given_y=sample_x_given_y,
        sample_y_given_x=sample_y_given_x,
        log_target=lambda x, y: log_target_density(x, y),
        var_names=("x", "y"),
    )

    # Metropolis-Hastings Random Walk
    mh_sampler = MetropolisHastingsSampler(
        log_target=log_target_2d,
        proposal_scale=np.array([0.2, 0.15]),
        var_names=["x", "y"],
        adaptive_scaling=True,
    )

    # MALA with automatic gradients
    try:
        import torch

        def log_target_torch(xy):
            """PyTorch-compatible log target function"""
            x, y = xy[0], xy[1]
            x_safe = torch.where(x > 0, x, torch.tensor(1e-10, dtype=x.dtype))
            log_prob = 2 * torch.log(x_safe) - x_safe * y**2 - y**2 + 2 * y - 4 * x_safe
            return torch.where(x > 0, log_prob, torch.tensor(-1e10, dtype=x.dtype))

        mala_sampler = MALAAutoGradSampler(
            log_target=log_target_torch, step_size=0.15, var_names=["x", "y"]
        )

        samplers = {
            "Independent MH": imh_sampler,
            "Gibbs Sampler": gibbs_sampler,
            "Metropolis-Hastings": mh_sampler,
            "MALA": mala_sampler,
        }

    except ImportError:
        print("Warning: PyTorch not available, skipping MALA sampler")
        samplers = {
            "Independent MH": imh_sampler,
            "Gibbs Sampler": gibbs_sampler,
            "Metropolis-Hastings": mh_sampler,
        }

    return samplers


def compare_methods():
    """Legacy comparison function - redirects to ArviZ version."""
    print("Note: Using improved ArviZ-based comparison framework")
    return compare_methods_arviz()


def plot_target_contours():
    """Plot contours of the target distribution for reference"""
    print("\nPlotting target distribution contours...")

    x = np.linspace(0.1, 3, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = target_density(X[j, i], Y[j, i])

    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=20, colors="black", alpha=0.5)
    plt.contourf(X, Y, Z, levels=20, cmap="viridis", alpha=0.7)
    plt.colorbar(label="Density")
    plt.title("Target Distribution: $x^2 e^{-xy^2 - y^2 + 2y - 4x}$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    print("Comparing Four MCMC Sampling Methods with ArviZ Diagnostics")
    print("=" * 60)
    print("Target: x² * exp(-x*y² - y² + 2*y - 4*x)")
    print("=" * 60)

    # Show target distribution
    plot_target_contours()

    # Run comprehensive ArviZ-based comparison
    results, efficiency_metrics, rankings = compare_methods_arviz()

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"Winner: {rankings['overall'][0]}")
    print("Results saved in 'results' dictionary for further analysis")

    # Optionally save results for later analysis
    # az.to_netcdf(results, "mcmc_comparison_results.nc")
