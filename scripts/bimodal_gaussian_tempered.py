"""Generate bimodal Gaussian distribution and its tempered version.

This script creates visualizations comparing the original bimodal Gaussian
probability density function with its tempered (raised to power gamma) version.
The tempered distribution is NOT normalized, showing the effect of tempering
on the density values directly.
"""

import matplotlib.pyplot as plt
import numpy as np


def bimodal_gaussian_pdf(
    x: np.ndarray, mu1: float, mu2: float, sigma: float
) -> np.ndarray:
    """Compute bimodal Gaussian PDF (mixture of two equal-weight Gaussians).

    Args:
        x: Points at which to evaluate the PDF
        mu1: Mean of first Gaussian
        mu2: Mean of second Gaussian
        sigma: Standard deviation (same for both)

    Returns:
        PDF values at x
    """
    pdf1 = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu1) / sigma) ** 2)
    pdf2 = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu2) / sigma) ** 2)
    return 0.5 * (pdf1 + pdf2)


def tempered_pdf(pdf: np.ndarray, gamma: float) -> np.ndarray:
    """Apply tempering to a PDF (raise to power gamma, without normalization).

    Args:
        pdf: Original PDF values
        gamma: Tempering parameter (0 < gamma <= 1)

    Returns:
        Tempered PDF values (unnormalized)
    """
    return pdf**gamma


def plot_bimodal_tempered(
    mu1: float = -1.0,
    mu2: float = 1.0,
    sigma: float = 0.5,
    gamma: float = 0.7,
    x_range: tuple[float, float] = (-3, 3),
    n_points: int = 1000,
) -> None:
    """Plot original and tempered bimodal Gaussian distributions.

    Args:
        mu1: Mean of first Gaussian
        mu2: Mean of second Gaussian
        sigma: Standard deviation
        gamma: Tempering parameter
        x_range: Range of x values to plot
        n_points: Number of points for plotting
    """
    # Generate x values
    x = np.linspace(x_range[0], x_range[1], n_points)

    # Compute original PDF
    pdf_original = bimodal_gaussian_pdf(x, mu1, mu2, sigma)

    # Compute tempered PDF (unnormalized)
    pdf_tempered = tempered_pdf(pdf_original, gamma)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        x, pdf_original, label="Original PDF", color="#CC6677", linewidth=2, alpha=0.7
    )
    plt.plot(
        x,
        pdf_tempered,
        label=f"PDF^{gamma} (normalized)",
        color="#117733",
        linewidth=2.5,
    )

    plt.xlabel("x", fontsize=12)
    plt.ylabel("density", fontsize=12)
    plt.title(
        f"Bimodal Gaussian: Original vs Tempered (γ={gamma})",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    output_path = "bimodal_gaussian_tempered.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    # Generate the plot with default parameters matching the reference image
    plot_bimodal_tempered(
        mu1=-1.0,
        mu2=1.0,
        sigma=0.2,
        gamma=0.1,
        x_range=(-3, 3),
        n_points=1000,
    )
