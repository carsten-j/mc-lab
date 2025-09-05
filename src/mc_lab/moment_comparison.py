import warnings
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


class MomentComparison:
    """
    Comprehensive framework for comparing theoretical moments from scipy distributions
    with empirical moments from sample data.
    """

    def __init__(self, scipy_dist, dist_params: Dict[str, Any], samples: np.ndarray):
        """
        Initialize moment comparison.

        Parameters:
        -----------
        scipy_dist : scipy.stats distribution
            The scipy distribution object (e.g., stats.norm, stats.cauchy)
        dist_params : dict
            Parameters for the distribution (e.g., {'loc': 0, 'scale': 1})
        samples : np.ndarray
            Empirical samples to compare against theoretical distribution
        """
        self.dist = scipy_dist(**dist_params)
        self.dist_name = scipy_dist.name
        self.dist_params = dist_params
        self.samples = np.asarray(samples)
        self.n_samples = len(samples)

        # Check if distribution has finite moments
        self._check_moment_existence()

    def _check_moment_existence(self):
        """Check which moments exist for the distribution."""
        self.moment_exists = {}

        # Special cases for known distributions
        if self.dist_name == "cauchy":
            self.moment_exists = {1: False, 2: False, 3: False, 4: False}
            warnings.warn("Cauchy distribution has no finite moments!")
        elif self.dist_name == "levy":
            self.moment_exists = {1: False, 2: False, 3: False, 4: False}
            warnings.warn("Lévy distribution has no finite moments!")
        elif self.dist_name == "pareto":
            # Pareto has finite moments only up to order alpha
            alpha = self.dist_params.get("b", 1)
            for k in range(1, 5):
                self.moment_exists[k] = k < alpha
        else:
            # For most distributions, first 4 moments exist
            self.moment_exists = {1: True, 2: True, 3: True, 4: True}

    def calculate_theoretical_moments(self, max_order: int = 4) -> Dict[str, float]:
        """
        Calculate theoretical moments from the scipy distribution.

        Parameters:
        -----------
        max_order : int
            Maximum moment order to calculate (default: 4)

        Returns:
        --------
        dict : Dictionary of theoretical moments
        """
        moments = {}

        try:
            # Get standard moments using stats method
            mean, var, skew, kurt = self.dist.stats(moments="mvsk")

            if self.moment_exists.get(1, True):
                moments["mean"] = float(mean)
            else:
                moments["mean"] = np.nan

            if self.moment_exists.get(2, True):
                moments["variance"] = float(var)
                moments["std"] = np.sqrt(float(var))
            else:
                moments["variance"] = np.nan
                moments["std"] = np.nan

            if self.moment_exists.get(3, True):
                moments["skewness"] = float(skew)
            else:
                moments["skewness"] = np.nan

            if self.moment_exists.get(4, True):
                moments["kurtosis"] = float(kurt)  # Excess kurtosis
            else:
                moments["kurtosis"] = np.nan

            # Calculate raw moments if they exist
            for k in range(1, min(max_order + 1, 5)):
                if self.moment_exists.get(k, True):
                    try:
                        moments[f"raw_moment_{k}"] = float(self.dist.moment(k))
                    except Exception as e:
                        warnings.warn(f"Error calculating raw moment {k}: {e}")
                        moments[f"raw_moment_{k}"] = np.nan
                else:
                    moments[f"raw_moment_{k}"] = np.nan

        except Exception as e:
            warnings.warn(f"Error calculating theoretical moments: {e}")
            moments = {
                "mean": np.nan,
                "variance": np.nan,
                "std": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
            }

        return moments

    def calculate_empirical_moments(
        self, max_order: int = 4, robust: bool = False
    ) -> Dict[str, float]:
        """
        Calculate empirical moments from sample data.

        Parameters:
        -----------
        max_order : int
            Maximum moment order to calculate
        robust : bool
            If True, use robust estimators (median, MAD, etc.)

        Returns:
        --------
        dict : Dictionary of empirical moments
        """
        moments = {}

        if robust:
            # Robust estimators
            moments["mean"] = np.median(self.samples)
            moments["variance"] = (
                stats.median_abs_deviation(self.samples, scale="normal") ** 2
            )
            moments["std"] = stats.median_abs_deviation(self.samples, scale="normal")

            # Robust skewness and kurtosis using quantiles
            q25, q50, q75 = np.percentile(self.samples, [25, 50, 75])
            moments["skewness"] = ((q75 - q50) - (q50 - q25)) / (q75 - q25)

            # Robust kurtosis using percentiles
            p10, p90 = np.percentile(self.samples, [10, 90])
            moments["kurtosis"] = (
                (p90 - p10) / (2 * 1.28 * moments["std"])
                if moments["std"] > 0
                else np.nan
            )
        else:
            # Standard moment estimators
            moments["mean"] = np.mean(self.samples)
            moments["variance"] = np.var(self.samples, ddof=1)
            moments["std"] = np.std(self.samples, ddof=1)
            moments["skewness"] = stats.skew(self.samples)
            moments["kurtosis"] = stats.kurtosis(
                self.samples, fisher=True
            )  # Excess kurtosis

        # Raw moments
        for k in range(1, min(max_order + 1, 5)):
            moments[f"raw_moment_{k}"] = np.mean(self.samples**k)

        return moments

    def bootstrap_confidence_intervals(
        self,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        moment_names: List[str] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate bootstrap confidence intervals for empirical moments.

        Parameters:
        -----------
        n_bootstrap : int
            Number of bootstrap samples
        confidence : float
            Confidence level (default: 0.95)
        moment_names : list
            Specific moments to calculate CIs for

        Returns:
        --------
        dict : Confidence intervals for each moment
        """
        if moment_names is None:
            moment_names = ["mean", "variance", "std", "skewness", "kurtosis"]

        bootstrap_moments = {name: [] for name in moment_names}

        for _ in range(n_bootstrap):
            # Resample with replacement
            boot_sample = np.random.choice(
                self.samples, size=self.n_samples, replace=True
            )

            # Calculate moments for bootstrap sample
            if "mean" in moment_names:
                bootstrap_moments["mean"].append(np.mean(boot_sample))
            if "variance" in moment_names:
                bootstrap_moments["variance"].append(np.var(boot_sample, ddof=1))
            if "std" in moment_names:
                bootstrap_moments["std"].append(np.std(boot_sample, ddof=1))
            if "skewness" in moment_names:
                bootstrap_moments["skewness"].append(stats.skew(boot_sample))
            if "kurtosis" in moment_names:
                bootstrap_moments["kurtosis"].append(
                    stats.kurtosis(boot_sample, fisher=True)
                )

        # Calculate confidence intervals
        alpha = 1 - confidence
        confidence_intervals = {}

        for name, values in bootstrap_moments.items():
            if values:
                lower = np.percentile(values, 100 * alpha / 2)
                upper = np.percentile(values, 100 * (1 - alpha / 2))
                confidence_intervals[name] = (lower, upper)

        return confidence_intervals

    def statistical_tests(
        self, theoretical_moments: Dict[str, float], empirical_moments: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform statistical tests comparing theoretical and empirical moments.

        Parameters:
        -----------
        theoretical_moments : dict
            Theoretical moments from distribution
        empirical_moments : dict
            Empirical moments from samples

        Returns:
        --------
        dict : Test statistics and p-values
        """
        test_results = {}

        # T-test for mean (if moments exist)
        if not np.isnan(theoretical_moments.get("mean", np.nan)):
            t_stat = (empirical_moments["mean"] - theoretical_moments["mean"]) / (
                empirical_moments["std"] / np.sqrt(self.n_samples)
            )
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=self.n_samples - 1))
            test_results["mean_test"] = {
                "t_statistic": t_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }

        # Chi-square test for variance (if moments exist)
        if not np.isnan(theoretical_moments.get("variance", np.nan)):
            chi2_stat = (
                (self.n_samples - 1)
                * empirical_moments["variance"]
                / theoretical_moments["variance"]
            )
            p_value = 2 * min(
                stats.chi2.cdf(chi2_stat, df=self.n_samples - 1),
                1 - stats.chi2.cdf(chi2_stat, df=self.n_samples - 1),
            )
            test_results["variance_test"] = {
                "chi2_statistic": chi2_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }

        # Z-test for skewness (large sample)
        if (
            not np.isnan(theoretical_moments.get("skewness", np.nan))
            and self.n_samples > 30
        ):
            se_skew = np.sqrt(6 / self.n_samples)
            z_stat = (
                empirical_moments["skewness"] - theoretical_moments["skewness"]
            ) / se_skew
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            test_results["skewness_test"] = {
                "z_statistic": z_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }

        # Z-test for kurtosis (large sample)
        if (
            not np.isnan(theoretical_moments.get("kurtosis", np.nan))
            and self.n_samples > 30
        ):
            se_kurt = np.sqrt(24 / self.n_samples)
            z_stat = (
                empirical_moments["kurtosis"] - theoretical_moments["kurtosis"]
            ) / se_kurt
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            test_results["kurtosis_test"] = {
                "z_statistic": z_stat,
                "p_value": p_value,
                "significant": p_value < 0.05,
            }

        return test_results

    def compare_moments(
        self,
        max_order: int = 4,
        robust: bool = False,
        bootstrap_ci: bool = True,
        n_bootstrap: int = 1000,
        plot: bool = True,
    ) -> Dict[str, Any]:
        """
        Complete moment comparison between theoretical and empirical distributions.

        Parameters:
        -----------
        max_order : int
            Maximum moment order to calculate
        robust : bool
            Use robust moment estimators
        bootstrap_ci : bool
            Calculate bootstrap confidence intervals
        n_bootstrap : int
            Number of bootstrap samples
        plot : bool
            Create visualization

        Returns:
        --------
        dict : Complete comparison results
        """
        # Calculate moments
        theoretical = self.calculate_theoretical_moments(max_order)
        empirical = self.calculate_empirical_moments(max_order, robust)

        # Calculate relative errors
        relative_errors = {}
        for key in ["mean", "variance", "std", "skewness", "kurtosis"]:
            if key in theoretical and key in empirical:
                if not np.isnan(theoretical[key]) and theoretical[key] != 0:
                    relative_errors[key] = abs(empirical[key] - theoretical[key]) / abs(
                        theoretical[key]
                    )
                elif not np.isnan(theoretical[key]):
                    relative_errors[key] = abs(empirical[key] - theoretical[key])
                else:
                    relative_errors[key] = np.nan

        # Bootstrap confidence intervals
        confidence_intervals = {}
        if bootstrap_ci:
            confidence_intervals = self.bootstrap_confidence_intervals(n_bootstrap)

        # Statistical tests
        test_results = self.statistical_tests(theoretical, empirical)

        # Create summary
        results = {
            "theoretical_moments": theoretical,
            "empirical_moments": empirical,
            "relative_errors": relative_errors,
            "confidence_intervals": confidence_intervals,
            "statistical_tests": test_results,
            "sample_size": self.n_samples,
            "distribution": self.dist_name,
            "parameters": self.dist_params,
        }

        # Visualization
        if plot:
            self._plot_comparison(theoretical, empirical, confidence_intervals)

        return results

    def _plot_comparison(
        self,
        theoretical: Dict[str, float],
        empirical: Dict[str, float],
        confidence_intervals: Dict[str, Tuple[float, float]],
    ):
        """Create visualization of moment comparison."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(
            f"Moment Comparison: {self.dist_name} Distribution (n={self.n_samples})",
            fontsize=14,
            fontweight="bold",
        )

        moment_names = ["mean", "variance", "std", "skewness", "kurtosis"]
        colors = ["blue", "green", "red", "purple", "orange"]

        for idx, (ax, moment, color) in enumerate(
            zip(axes.flat[:5], moment_names, colors)
        ):
            theo_val = theoretical.get(moment, np.nan)
            emp_val = empirical.get(moment, np.nan)

            if not np.isnan(theo_val) and not np.isnan(emp_val):
                # Bar plot
                x = np.arange(2)
                values = [theo_val, emp_val]
                bars = ax.bar(x, values, color=[color, "gray"], alpha=0.7)
                ax.set_xticks(x)
                ax.set_xticklabels(["Theoretical", "Empirical"])
                ax.set_ylabel(moment.capitalize())
                ax.set_title(f"{moment.capitalize()} Comparison")

                # Add confidence interval if available
                if moment in confidence_intervals:
                    ci_lower, ci_upper = confidence_intervals[moment]
                    ax.errorbar(
                        1,
                        emp_val,
                        yerr=[[emp_val - ci_lower], [ci_upper - emp_val]],
                        fmt="none",
                        color="black",
                        capsize=5,
                    )

                # Add value labels
                for bar, val in zip(bars, values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{val:.4f}",
                        ha="center",
                        va="bottom",
                    )

                # Add relative error
                if theo_val != 0:
                    rel_error = abs(emp_val - theo_val) / abs(theo_val) * 100
                    ax.text(
                        0.5,
                        0.95,
                        f"Rel. Error: {rel_error:.1f}%",
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Moment does not exist\nor cannot be calculated",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                    color="red",
                )
                ax.set_title(f"{moment.capitalize()} Comparison")

        # Q-Q plot in the last subplot
        ax = axes.flat[5]
        stats.probplot(self.samples, dist=self.dist, plot=ax)
        ax.set_title("Q-Q Plot")

        plt.tight_layout()
        plt.show()

    def create_report(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        Create a pandas DataFrame report of the comparison results.

        Parameters:
        -----------
        results : dict
            Results from compare_moments()

        Returns:
        --------
        pd.DataFrame : Formatted report
        """
        report_data = []

        moment_names = ["mean", "variance", "std", "skewness", "kurtosis"]

        for moment in moment_names:
            row = {
                "Moment": moment.capitalize(),
                "Theoretical": results["theoretical_moments"].get(moment, np.nan),
                "Empirical": results["empirical_moments"].get(moment, np.nan),
                "Relative Error": results["relative_errors"].get(moment, np.nan),
            }

            # Add confidence interval
            if moment in results["confidence_intervals"]:
                ci = results["confidence_intervals"][moment]
                row["95% CI"] = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            else:
                row["95% CI"] = "N/A"

            # Add test results
            test_key = f"{moment}_test"
            if test_key in results["statistical_tests"]:
                test = results["statistical_tests"][test_key]
                row["p-value"] = test["p_value"]
                row["Significant"] = test["significant"]
            else:
                row["p-value"] = np.nan
                row["Significant"] = "N/A"

            report_data.append(row)

        return pd.DataFrame(report_data)


def compare_distribution_moments(
    scipy_dist,
    dist_params: Dict[str, Any],
    samples: np.ndarray,
    robust: bool = False,
    bootstrap_ci: bool = True,
    n_bootstrap: int = 1000,
    plot: bool = False,
    create_report: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function for quick moment comparison.

    Parameters:
    -----------
    scipy_dist : scipy.stats distribution
        The scipy distribution object (e.g., stats.norm, stats.cauchy)
    dist_params : dict
        Parameters for the distribution (e.g., {'loc': 0, 'scale': 1})
    samples : array-like
        Empirical samples to compare
    robust : bool
        Use robust moment estimators
    bootstrap_ci : bool
        Calculate bootstrap confidence intervals
    n_bootstrap : int
        Number of bootstrap samples
    plot : bool
        Create visualization
    create_report : bool
        Generate pandas DataFrame report

    Returns:
    --------
    dict : Comparison results including report if requested

    Example:
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>>
    >>> # Generate samples from a normal distribution
    >>> true_samples = stats.norm.rvs(loc=5, scale=2, size=1000)
    >>>
    >>> # Compare with theoretical distribution
    >>> results = compare_distribution_moments(
    ...     scipy_dist=stats.norm,
    ...     dist_params={'loc': 5, 'scale': 2},
    ...     samples=true_samples,
    ...     plot=True
    ... )
    >>>
    >>> # Check if simulation matches theory
    >>> print(results['report'])
    """
    # Initialize comparison object
    comparator = MomentComparison(scipy_dist, dist_params, samples)

    # Run comparison
    results = comparator.compare_moments(
        robust=robust, bootstrap_ci=bootstrap_ci, n_bootstrap=n_bootstrap, plot=plot
    )

    # Create report if requested
    if create_report:
        results["report"] = comparator.create_report(results)
        print("\n" + "=" * 60)
        print("MOMENT COMPARISON REPORT")
        print("=" * 60)
        print(f"Distribution: {scipy_dist.name}")
        print(f"Parameters: {dist_params}")
        print(f"Sample Size: {len(samples)}")
        print("-" * 60)
        print(results["report"].to_string(index=False))
        print("=" * 60)

    return results


# Example usage for different distributions
if __name__ == "__main__":
    # Example 1: Normal distribution (all moments exist)
    print("\n### Example 1: Normal Distribution ###")
    normal_samples = stats.norm.rvs(loc=0, scale=1, size=1000, random_state=42)
    normal_results = compare_distribution_moments(
        scipy_dist=stats.norm,
        dist_params={"loc": 0, "scale": 1},
        samples=normal_samples,
        plot=False,  # Set to True to see plots
    )

    # Example 2: Cauchy distribution (no moments exist)
    print("\n### Example 2: Cauchy Distribution ###")
    cauchy_samples = stats.cauchy.rvs(loc=0, scale=1, size=1000, random_state=42)
    cauchy_results = compare_distribution_moments(
        scipy_dist=stats.cauchy,
        dist_params={"loc": 0, "scale": 1},
        samples=cauchy_samples,
        robust=True,  # Use robust estimators for heavy-tailed data
        plot=False,
    )

    # Example 3: Exponential distribution
    print("\n### Example 3: Exponential Distribution ###")
    exp_samples = stats.expon.rvs(scale=2, size=1000, random_state=42)
    exp_results = compare_distribution_moments(
        scipy_dist=stats.expon,
        dist_params={"scale": 2},
        samples=exp_samples,
        plot=False,
    )
