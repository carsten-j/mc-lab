import numpy as np
from scipy import stats


def distribution_comparison(sample1, sample2, alpha=0.05, print_result=True):
    """Complete framework for comparing two distributions."""

    # Preprocessing
    s1 = np.array(sample1)[~np.isnan(sample1)]
    s2 = np.array(sample2)[~np.isnan(sample2)]

    # Statistical tests
    ks_stat, ks_p = stats.ks_2samp(s1, s2)
    ad_result = stats.anderson_ksamp([s1, s2])
    res = stats.cramervonmises_2samp(s1, s2)

    # Results summary
    results = {
        "kolmogorov_smirnov": {"statistic": ks_stat, "p_value": ks_p},
        "anderson_darling": {
            "statistic": ad_result.statistic,
            "p_value": ad_result.pvalue,
        },
        "cramer_von_mises": {"statistic": res.statistic, "p_value": res.pvalue},
    }

    if print_result:
        print("Distribution Comparison Results:")
        print("=" * 80)
        print(
            f"{'Distribution Test':<25} {'Statistic':<12} {'P-value':<12} {'Significant at α={alpha}':<20}"
        )
        print("-" * 80)
        print(
            f"{'Kolmogorov-Smirnov':<25} {ks_stat:<12.6f} {ks_p:<12.6f} {'Yes' if ks_p < alpha else 'No':<20}"
        )
        print(
            f"{'Anderson-Darling':<25} {ad_result.statistic:<12.6f} {ad_result.pvalue:<12.6f} {'Yes' if ad_result.pvalue < alpha else 'No':<20}"
        )
        print(
            f"{'Cramér-von Mises':<25} {res.statistic:<12.6f} {res.pvalue:<12.6f} {'Yes' if res.pvalue < alpha else 'No':<20}"
        )
        print("=" * 80)

    return results
