import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Define bimodal distribution
def bimodal_pdf(x, mu1=-5, mu2=8, sigma1=1, sigma2=1, w1=0.5):
    """Bimodal Gaussian distribution with wide valley"""
    return w1 * norm.pdf(x, mu1, sigma1) + (1-w1) * norm.pdf(x, mu2, sigma2)

def bimodal_logpdf(x, mu1=-5, mu2=8, sigma1=1, sigma2=1, w1=0.5):
    """Log of bimodal distribution"""
    p1 = w1 * norm.pdf(x, mu1, sigma1)
    p2 = (1-w1) * norm.pdf(x, mu2, sigma2)
    return np.log(p1 + p2 + 1e-10)

# Metropolis-Hastings MCMC
def metropolis_hastings(n_iterations, initial_value, proposal_std=0.5):
    """
    Simple Metropolis-Hastings sampler
    """
    samples = np.zeros(n_iterations)
    samples[0] = initial_value
    accepted = 0
    
    for i in range(1, n_iterations):
        current = samples[i-1]
        
        # Propose new value
        proposed = current + np.random.normal(0, proposal_std)
        
        # Calculate acceptance ratio
        log_ratio = bimodal_logpdf(proposed) - bimodal_logpdf(current)
        
        # Accept or reject
        if np.log(np.random.random()) < log_ratio:
            samples[i] = proposed
            accepted += 1
        else:
            samples[i] = current
    
    acceptance_rate = accepted / n_iterations
    return samples, acceptance_rate

# Parameters
mu1, mu2 = -5, 8  # Wide valley between modes
sigma1, sigma2 = 1, 1
w1 = 0.5
n_iterations = 5000
initial_value = -8  # Start left of left mode
proposal_std = 0.5

# Run MCMC
np.random.seed(42)
samples, acceptance_rate = metropolis_hastings(n_iterations, initial_value, proposal_std)

# Calculate running mean (estimator convergence)
running_mean = np.cumsum(samples) / np.arange(1, n_iterations + 1)
true_mean = w1 * mu1 + (1-w1) * mu2

# Create visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Bimodal Distribution
ax1 = fig.add_subplot(gs[0, :])
x = np.linspace(-10, 12, 1000)
y = bimodal_pdf(x, mu1, mu2, sigma1, sigma2, w1)

ax1.plot(x, y, 'b-', linewidth=2.5, label='True Distribution')
ax1.fill_between(x, y, alpha=0.3)
ax1.axvline(mu1, color='red', linestyle='--', linewidth=2, label=f'Left Mode (μ={mu1})')
ax1.axvline(mu2, color='green', linestyle='--', linewidth=2, label=f'Right Mode (μ={mu2})')
ax1.axvline(initial_value, color='orange', linestyle=':', linewidth=2, label=f'Initial Value ({initial_value})')

# Shade the valley
valley_x = x[(x > -1) & (x < 5)]
valley_y = bimodal_pdf(valley_x, mu1, mu2, sigma1, sigma2, w1)
ax1.fill_between(valley_x, valley_y, alpha=0.5, color='gray', label='Wide Valley')

ax1.set_xlabel('x', fontsize=12, fontweight='bold')
ax1.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
ax1.set_title('Bimodal Gaussian Distribution with Wide Valley', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. MCMC Trace Plot
ax2 = fig.add_subplot(gs[1, :])
iterations = np.arange(n_iterations)
ax2.plot(iterations, samples, 'steelblue', linewidth=0.5, alpha=0.7)
ax2.axhline(mu1, color='red', linestyle='--', linewidth=2, label=f'Left Mode (μ={mu1})')
ax2.axhline(mu2, color='green', linestyle='--', linewidth=2, label=f'Right Mode (μ={mu2})')
ax2.axhline(true_mean, color='purple', linestyle='-.', linewidth=2, label=f'True Mean ({true_mean:.2f})')

# Add shaded region for valley
ax2.axhspan(-1, 5, alpha=0.2, color='gray', label='Valley Region')

ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax2.set_ylabel('Sample Value', fontsize=12, fontweight='bold')
ax2.set_title(f'MCMC Trace Plot (Acceptance Rate: {acceptance_rate:.2%})', 
              fontsize=14, fontweight='bold')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, n_iterations)

# 3. Histogram of Samples (Left)
ax3 = fig.add_subplot(gs[2, 0])
ax3.hist(samples, bins=50, density=True, alpha=0.6, color='steelblue', 
         edgecolor='black', label='MCMC Samples')
ax3.plot(x, y, 'b-', linewidth=2.5, label='True Distribution')
ax3.axvline(mu1, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax3.axvline(mu2, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax3.set_xlabel('x', fontsize=12, fontweight='bold')
ax3.set_ylabel('Density', fontsize=12, fontweight='bold')
ax3.set_title('Sample Distribution vs True Distribution', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Add text annotation
ax3.text(0.05, 0.95, f'Samples in left mode: {np.sum(samples < 0)}\nSamples in right mode: {np.sum(samples > 5)}',
         transform=ax3.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Convergence of Estimator (Right)
ax4 = fig.add_subplot(gs[2, 1])
ax4.plot(iterations, running_mean, 'darkblue', linewidth=2, label='Running Mean')
ax4.axhline(true_mean, color='purple', linestyle='-.', linewidth=2.5, 
            label=f'True Mean ({true_mean:.2f})')
ax4.axhline(np.mean(samples), color='orange', linestyle=':', linewidth=2.5,
            label=f'MCMC Estimate ({np.mean(samples):.2f})')

ax4.fill_between(iterations, 
                  running_mean - 2*np.std(samples)/np.sqrt(iterations+1),
                  running_mean + 2*np.std(samples)/np.sqrt(iterations+1),
                  alpha=0.3, color='blue', label='95% CI (approx)')

ax4.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax4.set_ylabel('Estimated Mean', fontsize=12, fontweight='bold')
ax4.set_title('Convergence of Mean Estimator', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, n_iterations)

# Add bias annotation
final_bias = np.mean(samples) - true_mean
ax4.text(0.95, 0.05, f'Final Bias: {final_bias:.3f}\nStd Dev: {np.std(samples):.3f}',
         transform=ax4.transAxes, fontsize=10, verticalalignment='bottom',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.suptitle('MCMC Failure in Bimodal Distribution: Trapped in Local Mode', 
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.show()

# Print summary statistics
print("="*60)
print("MCMC SAMPLING SUMMARY")
print("="*60)
print(f"True Distribution Parameters:")
print(f"  Left Mode (μ₁):  {mu1}")
print(f"  Right Mode (μ₂): {mu2}")
print(f"  Weights:         {w1:.1f}, {1-w1:.1f}")
print(f"  True Mean:       {true_mean:.3f}")
print(f"\nMCMC Results:")
print(f"  Initial Value:   {initial_value}")
print(f"  Iterations:      {n_iterations}")
print(f"  Acceptance Rate: {acceptance_rate:.2%}")
print(f"\nSample Statistics:")
print(f"  Sample Mean:     {np.mean(samples):.3f}")
print(f"  Sample Std:      {np.std(samples):.3f}")
print(f"  Bias:            {np.mean(samples) - true_mean:.3f}")
print(f"\nMode Coverage:")
print(f"  Left mode (x < 0):  {np.sum(samples < 0)} samples ({100*np.sum(samples < 0)/n_iterations:.1f}%)")
print(f"  Valley (-1 < x < 5): {np.sum((samples > -1) & (samples < 5))} samples ({100*np.sum((samples > -1) & (samples < 5))/n_iterations:.1f}%)")
print(f"  Right mode (x > 5): {np.sum(samples > 5)} samples ({100*np.sum(samples > 5)/n_iterations:.1f}%)")
print("="*60)