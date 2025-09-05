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
        print("=" * 35)
        print("Kolmogorov-Smirnov Test:")
        print(f"  Statistic: {ks_stat:.6f}")
        print(f"  P-value: {ks_p:.6f}")
        print(f"  Significant at α={alpha}: {'Yes' if ks_p < alpha else 'No'}")
        print()
        print("Anderson-Darling Test:")
        print(f"  Statistic: {ad_result.statistic:.6f}")
        print(f"  P-value: {ad_result.pvalue:.6f}")
        print(
            f"  Significant at α={alpha}: {'Yes' if ad_result.pvalue < alpha else 'No'}"
        )
        print()
        print("Cramér-von Mises Test:")
        print(f"  Statistic: {res.statistic:.6f}")
        print(f"  P-value: {res.pvalue:.6f}")
        print(f"  Significant at α={alpha}: {'Yes' if res.pvalue < alpha else 'No'}")
        print("=" * 35)

    return results
