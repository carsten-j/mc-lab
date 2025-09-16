import arviz as az
import numpy as np
import torch
import torch.autograd as autograd
import tqdm


class ParallelMALA:
    """
    Parallel Metropolis-adjusted Langevin Algorithm using PyTorch for automatic differentiation.
    Optimized for GPU performance with batched operations.
    """

    def __init__(self, log_target, step_size=0.1, n_chains=4, device="cpu"):
        """
        Initialize Parallel PyTorch-based MALA sampler.

        Parameters:
        -----------
        log_target : callable
            Function that computes log probability of target distribution.
            Should accept a PyTorch tensor of shape (batch_size, dim) and return (batch_size,).
        step_size : float or array-like
            Step size parameter (epsilon). Can be different for each chain.
        n_chains : int
            Number of parallel chains to run
        device : str
            Device to run computations on ('cpu' or 'cuda')
        """
        self.log_target_orig = log_target
        self.n_chains = n_chains
        self.device = torch.device(device)

        # Handle step size for multiple chains - keep on GPU if using CUDA
        if np.isscalar(step_size):
            self.step_sizes = torch.full(
                (n_chains,), step_size, device=self.device, dtype=torch.float32
            )
        else:
            assert len(step_size) == n_chains, (
                "Step size array must match number of chains"
            )
            self.step_sizes = torch.tensor(
                step_size, device=self.device, dtype=torch.float32
            )

    def log_target_batch(self, x_batch):
        """
        Compute log target for a batch of points.

        Parameters:
        -----------
        x_batch : torch.Tensor shape (batch_size, dim)
            Batch of points

        Returns:
        --------
        log_probs : torch.Tensor shape (batch_size,)
            Log probabilities
        """
        if not isinstance(x_batch, torch.Tensor):
            x_batch = torch.tensor(x_batch, device=self.device, dtype=torch.float32)

        if x_batch.device != self.device:
            x_batch = x_batch.to(self.device)

        return self.log_target_orig(x_batch, self.device)

    def compute_gradient_batch(self, x_batch):
        """
        Compute gradients for a batch of points using vectorized operations.

        Parameters:
        -----------
        x_batch : torch.Tensor shape (n_chains, dim)
            Batch of points

        Returns:
        --------
        grads : torch.Tensor shape (n_chains, dim)
            Gradients of log target at each point
        """
        if not isinstance(x_batch, torch.Tensor):
            x_batch = torch.tensor(x_batch, device=self.device, dtype=torch.float32)
        else:
            x_batch = x_batch.clone().detach().to(self.device)

        x_batch.requires_grad_(True)

        # Compute log probabilities for all chains at once
        log_probs = self.log_target_batch(x_batch)

        # Compute gradients for all chains at once
        grads = autograd.grad(
            outputs=log_probs.sum(),  # Sum to get scalar for backward
            inputs=x_batch,
            create_graph=False,
        )[0]

        return grads.detach()

    def propose_batch_gpu(self, x_batch):
        """
        Generate proposals using Langevin dynamics for all chains on GPU.

        Parameters:
        -----------
        x_batch : torch.Tensor shape (n_chains, dim)
            Current states

        Returns:
        --------
        x_proposed : torch.Tensor shape (n_chains, dim)
            Proposed states
        grads : torch.Tensor shape (n_chains, dim)
            Gradients at current states (for reuse in density calculation)
        """
        # Compute gradients (stays on GPU)
        grads = self.compute_gradient_batch(x_batch)

        # Compute proposal means
        step_sizes_expanded = self.step_sizes.unsqueeze(1)
        means = x_batch + 0.5 * step_sizes_expanded**2 * grads

        # Generate noise on GPU
        noise = torch.randn_like(x_batch, device=self.device)

        # Return proposals and gradients
        return means + step_sizes_expanded * noise, grads

    def log_proposal_density_batch(self, x_new, x_old, grads_old, step_sizes):
        """
        Compute log proposal densities for batch on GPU.

        Parameters:
        -----------
        x_new : torch.Tensor shape (n_chains, dim)
            New states
        x_old : torch.Tensor shape (n_chains, dim)
            Old states
        grads_old : torch.Tensor shape (n_chains, dim)
            Pre-computed gradients at old states
        step_sizes : torch.Tensor shape (n_chains,)
            Step sizes for each chain
        """
        # Compute means using pre-computed gradients
        step_sizes_expanded = step_sizes.unsqueeze(1)
        means = x_old + 0.5 * step_sizes_expanded**2 * grads_old

        # Compute log densities (multivariate normal)
        diff = x_new - means
        dim = x_new.shape[1]

        # Log normalizing constant
        log_norm = -0.5 * dim * torch.log(2 * torch.tensor(np.pi, device=self.device))
        log_norm = log_norm - dim * torch.log(step_sizes)

        # Quadratic term
        quad_term = -0.5 * torch.sum(diff**2, dim=1) / step_sizes**2

        return log_norm + quad_term

    def accept_reject_batch_gpu(self, x_current, x_proposed, grads_current):
        """
        Perform accept/reject step for all chains on GPU.

        Parameters:
        -----------
        x_current : torch.Tensor shape (n_chains, dim)
            Current states
        x_proposed : torch.Tensor shape (n_chains, dim)
            Proposed states
        grads_current : torch.Tensor shape (n_chains, dim)
            Pre-computed gradients at current states

        Returns:
        --------
        x_next : torch.Tensor shape (n_chains, dim)
            Next states
        accepted : torch.Tensor shape (n_chains,)
            Boolean acceptance indicators
        alphas : torch.Tensor shape (n_chains,)
            Acceptance probabilities
        log_probs : torch.Tensor shape (n_chains,)
            Log probabilities of next states
        grads_next : torch.Tensor shape (n_chains, dim)
            Gradients at next states
        """
        # Compute gradients for proposed states (outside no_grad context)
        grads_proposed = self.compute_gradient_batch(x_proposed)

        with torch.no_grad():
            # Compute log probabilities
            log_current = self.log_target_batch(x_current)
            log_proposed = self.log_target_batch(x_proposed)

            # Compute proposal densities using pre-computed gradients
            log_q_forward = self.log_proposal_density_batch(
                x_proposed, x_current, grads_current, self.step_sizes
            )
            log_q_reverse = self.log_proposal_density_batch(
                x_current, x_proposed, grads_proposed, self.step_sizes
            )

            # Compute acceptance ratios
            log_alpha = log_proposed - log_current + log_q_reverse - log_q_forward
            alphas = torch.minimum(torch.ones_like(log_alpha), torch.exp(log_alpha))

            # Generate uniform random numbers on GPU
            u = torch.rand(self.n_chains, device=self.device)
            accepted = u < alphas

            # Select next states and gradients
            accepted_expanded = accepted.unsqueeze(1)
            x_next = torch.where(accepted_expanded, x_proposed, x_current)
            grads_next = torch.where(accepted_expanded, grads_proposed, grads_current)
            log_probs = torch.where(accepted, log_proposed, log_current)

        return x_next, accepted, alphas, log_probs, grads_next

    def sample(
        self,
        x0,
        n_samples,
        burn_in=0,
        thin=1,
        verbose=True,
        var_names=None,
        coords=None,
        dims=None,
        attrs=None,
    ):
        """
        Generate samples using parallel MALA chains with GPU optimization.
        """
        # Prepare initial states on GPU
        x0 = np.asarray(x0, dtype=np.float32)
        if x0.ndim == 1:
            x0_base = x0
            x0 = np.tile(x0_base, (self.n_chains, 1))
            x0 += np.random.normal(0, 0.01, x0.shape)

        assert x0.shape[0] == self.n_chains, (
            "Initial states must match number of chains"
        )

        dim = x0.shape[1]
        total_iterations = burn_in + n_samples * thin

        # Move initial states to GPU
        x_current = torch.tensor(x0, device=self.device, dtype=torch.float32)

        # Pre-allocate storage for better performance
        n_saved = (total_iterations - burn_in + thin - 1) // thin
        samples = torch.zeros(
            (n_saved, self.n_chains, dim), device="cpu", dtype=torch.float32
        )
        log_probs = torch.zeros(
            (n_saved, self.n_chains), device="cpu", dtype=torch.float32
        )
        acceptance_probs = torch.zeros(
            (n_saved, self.n_chains), device="cpu", dtype=torch.float32
        )

        n_accepted = torch.zeros(self.n_chains, device=self.device)
        saved_idx = 0

        # Progress bar
        iterator = range(total_iterations)
        if verbose:
            iterator = tqdm.tqdm(iterator, desc="Sampling")

        # Initialize gradients for current state
        grads_current = self.compute_gradient_batch(x_current)

        for i in iterator:
            # Propose new states (all operations on GPU)
            x_proposed, _ = self.propose_batch_gpu(x_current)

            # Accept or reject (all operations on GPU)
            x_current, accepted, alphas, current_log_probs, grads_current = (
                self.accept_reject_batch_gpu(x_current, x_proposed, grads_current)
            )

            n_accepted += accepted.float()

            # Store samples if past burn-in and at thinning interval
            if i >= burn_in and (i - burn_in) % thin == 0:
                # Transfer to CPU only when storing
                samples[saved_idx] = x_current.cpu()
                log_probs[saved_idx] = current_log_probs.cpu()
                acceptance_probs[saved_idx] = alphas.cpu()
                saved_idx += 1

            # Update progress bar
            if verbose and (i + 1) % 100 == 0:
                mean_acc_rate = (n_accepted.mean() / (i + 1)).item()
                iterator.set_postfix({"acc_rate": f"{mean_acc_rate:.3f}"})

        # Convert to numpy and reshape
        samples = samples.numpy().transpose(1, 0, 2)  # (n_chains, n_samples, dim)
        log_probs = log_probs.numpy().T  # (n_chains, n_samples)
        acceptance_probs = acceptance_probs.numpy().T  # (n_chains, n_samples)

        # Create variable names if not provided
        if var_names is None:
            if dim == 1:
                var_names = ["x"]
            else:
                var_names = [f"x_{i}" for i in range(dim)]

        # Prepare data dictionary for ArviZ
        posterior_dict = {}

        if dim == 1:
            posterior_dict[var_names[0]] = samples[:, :, 0]
        else:
            if len(var_names) == 1:
                posterior_dict[var_names[0]] = samples
            else:
                for i, var_name in enumerate(var_names):
                    posterior_dict[var_name] = samples[:, :, i]

        # Sample statistics
        sample_stats = {
            "lp": log_probs,
            "acceptance_rate": acceptance_probs,
            "n_steps": np.ones_like(log_probs),
            "diverging": np.zeros_like(log_probs, dtype=bool),
        }

        # Create attributes dictionary
        if attrs is None:
            attrs = {}

        final_acc_rate = (n_accepted.cpu().numpy() / total_iterations).mean()
        attrs.update(
            {
                "sampler": "MALA",
                "step_size": self.step_sizes.cpu().numpy().tolist(),
                "n_chains": self.n_chains,
                "n_samples": n_samples,
                "burn_in": burn_in,
                "thin": thin,
                "acceptance_rate": float(final_acc_rate),
                "device": str(self.device),
            }
        )

        # Create InferenceData object
        idata = az.from_dict(
            posterior=posterior_dict,
            sample_stats=sample_stats,
            coords=coords,
            dims=dims,
            attrs=attrs,
        )

        return idata


# Example usage function
def example_usage(device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Example showing how to use the ParallelMALA sampler with ArviZ.
    """
    import time

    import matplotlib.pyplot as plt

    print(f"Running on device: {device}")

    # Define a batched target distribution (2D Gaussian)
    def log_target(x_batch, device):
        """
        Batched log probability of a 2D Gaussian distribution.

        Parameters:
        -----------
        x_batch : torch.Tensor shape (batch_size, 2)
            Batch of input tensors
        device : torch.device
            Device to place tensors on

        Returns:
        --------
        log_probs : torch.Tensor shape (batch_size,)
            Log probabilities for each point
        """
        # Create tensors on the correct device
        mean = torch.zeros(2, device=device)
        cov = torch.tensor([[1.0, 0.5], [0.5, 1.0]], device=device, dtype=torch.float32)
        inv_cov = torch.inverse(cov)

        # Compute for batch
        diff = x_batch - mean
        # Batch matrix multiplication
        result = -0.5 * torch.sum((diff @ inv_cov) * diff, dim=1)
        return result

    # Create sampler
    sampler = ParallelMALA(log_target, step_size=0.5, n_chains=4, device=device)

    # Time the sampling
    start_time = time.time()

    # Sample
    x0 = np.array([0.0, 0.0])
    idata = sampler.sample(
        x0, n_samples=2000, burn_in=500, thin=2, var_names=["param1", "param2"]
    )

    end_time = time.time()
    print(f"\nSampling took {end_time - start_time:.2f} seconds")

    # Use ArviZ for diagnostics
    print("\n=== Summary Statistics ===")
    print(az.summary(idata))

    # Check convergence
    print("\n=== Convergence Diagnostics ===")
    print("R-hat values:")
    print(az.rhat(idata))
    print("\nEffective Sample Sizes:")
    print(az.ess(idata))

    # Plot traces
    az.plot_trace(idata)
    plt.suptitle("Trace Plots")
    plt.tight_layout()
    plt.show()

    return idata


if __name__ == "__main__":
    # Automatically use CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    idata = example_usage(device=device)
