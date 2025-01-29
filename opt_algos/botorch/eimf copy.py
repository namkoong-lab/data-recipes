"""
Plan:
1. Implement surrogate model with [w1 ... w5, model_size, step]
2. Import cost model with [model_size, step] to get flops
3. Implement EI acquisition function
4. Make it cost aware by dividing EI by flops
5. Implement optimization loop
"""

import os
import sys
import torch
import logging
import traceback
from pathlib import Path
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement, AcquisitionFunction, PosteriorMean
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.models.kernels.exponential_decay import ExponentialDecayKernel
from gpytorch.kernels import MaternKernel, ScaleKernel, ProductKernel
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan
from gpytorch.distributions import MultivariateNormal
from torch.nn import ModuleList
from torch import Tensor
from typing import Optional
import matplotlib.pyplot as plt
from datetime import datetime
import wandb
import numpy as np

# Add the root directory to the path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from opt_algos.data_model import get_flops
from opt_algos.benchmarks import NewDataModelBenchmark

# Global settings
GPU_ID = 2  # Change this to use different GPU
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

tkwargs = {
    "dtype": torch.double,
    "device": torch.device(DEVICE),
}

SMOKE_TEST = os.environ.get("SMOKE_TEST")

# Valid model sizes in millions of parameters
VALID_MODEL_SIZES = [20, 60, 150, 300, 500, 700, 1000]

# Valid step sizes (100 to 19600 in multiples of 100)
VALID_STEPS = list(range(100, 19501, 100))


def get_nearest_model_size(size):
    """Get the nearest valid model size."""
    return min(VALID_MODEL_SIZES, key=lambda x: abs(x - size))


def get_nearest_step(step):
    """Get the nearest valid step size (multiple of 100)."""
    return min(VALID_STEPS, key=lambda x: abs(x - step))


def generate_random_valid_points(n, **tkwargs):
    """Generate n random points with valid model sizes and steps.
    Returns tensor of shape [n, 7]
    """
    # Random weights between -3 and 3 [n, 5]
    weights = torch.rand(n, 5, **tkwargs) * 6.0 - 3.0

    # Random model sizes from valid sizes [n, 1]
    model_size_indices = torch.randint(0, len(VALID_MODEL_SIZES), (n,))
    model_sizes = torch.tensor(
        [VALID_MODEL_SIZES[i] for i in model_size_indices], **tkwargs
    ).unsqueeze(-1)

    # Random steps from valid steps [n, 1]
    step_indices = torch.randint(0, len(VALID_STEPS), (n,))
    steps = torch.tensor([VALID_STEPS[i] for i in step_indices], **tkwargs).unsqueeze(
        -1
    )

    # Combine all parameters to [n, 7]
    points = torch.cat((weights, model_sizes, steps), dim=1)
    return points


def compute_cost(X):
    """
    Compute the cost (FLOPS) for input configurations.
    Args:
        X: tensor of shape (batch_size, 7) or (batch_size, q, 7) containing
           [w1...w5, model_size, step] for each point
    Returns:
        tensor of costs with shape (batch_size, 1) or (batch_size, q, 1)
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, **tkwargs)

    # Handle both 2D and 3D inputs
    orig_shape = X.shape
    if len(orig_shape) == 3:
        X = X.reshape(-1, orig_shape[-1])

    costs = []
    for x in X:
        model_size = x[-2].item()  # Second to last column
        step = int(x[-1].item())  # Last column, convert to int
        # Round step to nearest multiple of 100
        step = get_nearest_step(step)
        try:
            cost = get_flops(model_size, step)
            costs.append(cost)
        except (ValueError, TypeError) as e:
            print(
                f"Warning: Error computing FLOPS for model_size={model_size}, step={step}: {e}"
            )
            # Return a high cost for invalid configurations
            costs.append(1e12)

    costs = torch.tensor(costs, **tkwargs)

    # Reshape back to original shape if needed
    if len(orig_shape) == 3:
        costs = costs.reshape(orig_shape[0], orig_shape[1])

    return costs.unsqueeze(-1)  # Shape: (batch_size, 1) or (batch_size, q, 1)


def project_to_discrete(X):
    """Project continuous X to discrete valid values for model size and step.
    Args:
        X: tensor of shape (batch_size, d) or (batch_size, q, d)
    Returns:
        tensor of same shape as input with discrete values for model size and step
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, **tkwargs)

    # Create a copy to avoid modifying the input
    X_discrete = X.clone()

    # Handle both 2D and 3D inputs
    if len(X.shape) == 2:
        # Project model sizes (index -2)
        X_discrete[..., -2] = torch.tensor(
            [get_nearest_model_size(x.item()) for x in X_discrete[..., -2]], **tkwargs
        )

        # Project steps (index -1)
        X_discrete[..., -1] = torch.tensor(
            [get_nearest_step(x.item()) for x in X_discrete[..., -1]], **tkwargs
        )
    else:  # 3D input
        # Project model sizes (index -2)
        X_discrete[..., -2] = torch.tensor(
            [get_nearest_model_size(x.item()) for x in X_discrete[..., -2].flatten()],
            **tkwargs,
        ).reshape(X_discrete[..., -2].shape)

        # Project steps (index -1)
        X_discrete[..., -1] = torch.tensor(
            [get_nearest_step(x.item()) for x in X_discrete[..., -1].flatten()],
            **tkwargs,
        ).reshape(X_discrete[..., -1].shape)

    return X_discrete


def setup_logging(run_name):
    """Set up logging configuration."""
    # Create results directory if it doesn't exist
    log_dir = Path("results") / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_file = log_dir / "optimization.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    return log_dir


class CostAwareEI(AcquisitionFunction):
    """Expected Improvement per unit cost (EIpu) that evaluates at target fidelity."""

    def __init__(self, model, best_f, alpha=1.0, reference_size=1000, reference_steps=19600):
        """Initialize with model and best observed value."""
        super().__init__(model)
        self.ei = ExpectedImprovement(model, best_f=best_f)
        self.alpha = alpha
        self.reference_size = reference_size
        self.reference_steps = reference_steps

    def forward(self, X):
        """
        Evaluate EI/cost for the given input, always projecting to target fidelity.
        Args:
            X: tensor of shape (batch_size, q, d)
        Returns:
            tensor of shape (batch_size, q)
        """
        logging.debug(f"CostAwareEI: Input X shape: {X.shape}")

        # Project X to valid discrete values before computing EI
        X_discrete = project_to_discrete(X)  # Maintains input shape
        logging.debug(f"CostAwareEI: X_discrete shape after projection: {X_discrete.shape}")

        # Project to target fidelity for EI evaluation
        X_target = X_discrete.clone()
        X_target[..., -2] = self.reference_size  # Set model size to 1B
        X_target[..., -1] = self.reference_steps  # Set steps to max
        logging.debug(f"CostAwareEI: X_target shape: {X_target.shape}")

        # Evaluate EI at target fidelity
        ei_values = self.ei(X_target)  # Shape: (batch_size, q)
        logging.debug(f"CostAwareEI: ei_values shape: {ei_values.shape}")

        # But use actual fidelity cost for denominator
        costs = compute_cost(X_discrete)  # Shape: (batch_size, q, 1)
        logging.debug(f"CostAwareEI: costs shape: {costs.shape}")

        # Avoid division by zero
        eps = torch.finfo(costs.dtype).eps
        costs = costs.clamp_min(eps)

        # Compute EI per unit cost
        costs = costs.squeeze(-1)  # Shape: (batch_size, q)
        if len(ei_values.shape) == 1:
            ei_values = ei_values.unsqueeze(-1)  # Add q dimension if missing

        eipu = ei_values / costs**self.alpha
        logging.debug(f"CostAwareEI: eipu shape: {eipu.shape}")

        return eipu.squeeze(-1)  # Return shape (batch_size,) for q=1


class EIMultiFidelityOptimizer:
    def __init__(self, metric_index=4):  # Default to Stack Exchange Cross Entropy
        # Define bounds for w1...w5, model_size, step
        self.bounds = torch.tensor(
            [
                [-3.0, -3.0, -3.0, -3.0, -3.0, 20.0, 100.0],  # lower bounds
                [3.0, 3.0, 3.0, 3.0, 3.0, 1000.0, 19500.0],  # upper bounds
            ],
            **tkwargs,
        )

        # Initialize the data model benchmark
        self.metric_index = metric_index
        self.problem = NewDataModelBenchmark(metric_index=metric_index, device=DEVICE)
        print(f"Optimizing for metric: {METRIC_NAMES[metric_index]}")

        # Evaluate reference point (1B model at max steps)
        self.reference_size = 1000  # 1B parameters
        self.reference_steps = 19600
        uniform_weights = [0.2] * 5  # Equal mixture
        self.reference_value = self.problem._raw_func(
            self.reference_size,
            self.reference_steps / 100,  # Divide by 100 as per _raw_func
            uniform_weights,
        )
        logging.info(f"\nReference point (1B model, {self.reference_steps} steps):")
        logging.info(f"Value: {self.reference_value:.4f}")
        logging.info(f"Uniform mixture weights: {uniform_weights}")

        # Initialize history tracking
        self.reset_history()

    def reset_history(self):
        """Reset optimization history."""
        self.history = {
            "train_x": None,
            "train_obj": None,
            "costs": [],
            "best_values": [],
            "best_value": float("inf"),  # Start with inf for minimization
            "best_point": None,
            "best_cost": None,
            "cumulative_costs": [],
            "recommended_values": [],  # Track values of recommendations at target settings
            "recommended_points": [],  # Track recommended points
        }

    def update_history(self, new_x, new_obj, new_cost):
        """Update optimization history with new point."""
        # First evaluation
        if self.history["train_x"] is None:
            self.history["train_x"] = new_x
            self.history["train_obj"] = new_obj
        else:
            self.history["train_x"] = torch.cat([self.history["train_x"], new_x])
            self.history["train_obj"] = torch.cat([self.history["train_obj"], new_obj])

        # Update costs
        self.history["costs"].append(new_cost.item())
        self.history["cumulative_costs"].append(sum(self.history["costs"]))

        # Update best values (remember we negated for maximization, so best_value should be negated back)
        current_best = -new_obj.item()  # Convert back to original scale
        if current_best < self.history.get(
            "best_value", float("inf")
        ):  # Use inf for minimization
            self.history["best_value"] = current_best
            self.history["best_point"] = new_x
            self.history["best_cost"] = new_cost.item()

        self.history["best_values"].append(self.history["best_value"])

        # Log to wandb (use original scale)
        wandb.log(
            {
                "iteration": len(self.history["costs"]) - 1,
                "current_value": current_best,  # Original scale
                "best_value": self.history["best_value"],  # Original scale
                "current_cost": new_cost.item(),
                "cumulative_cost": self.history["cumulative_costs"][-1],
                "current_model_size": new_x[0, 5].item(),
                "current_step": new_x[0, 6].item(),
            }
        )

    def generate_initial_data(self, n=16):
        """Generate initial training data."""
        # Generate points with valid discrete values
        train_x = generate_random_valid_points(n, **tkwargs)

        # Get actual objective values from data model and negate for maximization
        train_obj = torch.tensor(
            [
                -self.problem._raw_func(  # Negate for maximization
                    x[5].item(),  # model size
                    x[6].item() / 100,  # step
                    x[:5].tolist(),  # weights
                )
                for x in train_x
            ],
            **tkwargs,
        ).reshape(-1, 1)

        self.history["train_x"] = train_x
        self.history["train_obj"] = train_obj

        return train_x, train_obj

    def initialize_model(self, train_x, train_obj):
        """Initialize the GP model."""
        model = DataMixtureMultiFidelityGP(
            train_x, train_obj,
            outcome_transform=Standardize(m=1)
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        
        # Log initial kernel parameters
        logging.debug("\nInitial kernel parameters:")
        logging.debug("Mixture kernel (Matern):")
        logging.debug(f"  lengthscales: {model.covar_module.base_kernel.mixture_kernel.lengthscale.detach().cpu().numpy()}")
        
        logging.debug("\nModel size kernel (ExponentialDecay):")
        logging.debug(f"  lengthscale: {model.covar_module.base_kernel.size_kernel.lengthscale.detach().cpu().numpy()}")
        logging.debug(f"  offset: {model.covar_module.base_kernel.size_kernel.offset.detach().cpu().numpy()}")
        logging.debug(f"  power: {model.covar_module.base_kernel.size_kernel.power.detach().cpu().numpy()}")
        
        logging.debug("\nSteps kernel (ExponentialDecay):")
        logging.debug(f"  lengthscale: {model.covar_module.base_kernel.step_kernel.lengthscale.detach().cpu().numpy()}")
        logging.debug(f"  offset: {model.covar_module.base_kernel.step_kernel.offset.detach().cpu().numpy()}")
        logging.debug(f"  power: {model.covar_module.base_kernel.step_kernel.power.detach().cpu().numpy()}")
        
        logging.debug(f"\nOutputscale: {model.covar_module.outputscale.detach().cpu().numpy()}")
        
        # Fit the model
        fit_gpytorch_mll(mll)
        
        # Log fitted kernel parameters
        logging.debug("\nFitted kernel parameters:")
        logging.debug("Mixture kernel (Matern):")
        logging.debug(f"  lengthscales: {model.covar_module.base_kernel.mixture_kernel.lengthscale.detach().cpu().numpy()}")
        
        logging.debug("\nModel size kernel (ExponentialDecay):")
        logging.debug(f"  lengthscale: {model.covar_module.base_kernel.size_kernel.lengthscale.detach().cpu().numpy()}")
        logging.debug(f"  offset: {model.covar_module.base_kernel.size_kernel.offset.detach().cpu().numpy()}")
        logging.debug(f"  power: {model.covar_module.base_kernel.size_kernel.power.detach().cpu().numpy()}")
        
        logging.debug("\nSteps kernel (ExponentialDecay):")
        logging.debug(f"  lengthscale: {model.covar_module.base_kernel.step_kernel.lengthscale.detach().cpu().numpy()}")
        logging.debug(f"  offset: {model.covar_module.base_kernel.step_kernel.offset.detach().cpu().numpy()}")
        logging.debug(f"  power: {model.covar_module.base_kernel.step_kernel.power.detach().cpu().numpy()}")
        
        logging.debug(f"\nOutputscale: {model.covar_module.outputscale.detach().cpu().numpy()}")
        
        # Log training data statistics
        logging.debug("\nTraining data statistics:")
        logging.debug(f"Number of points: {len(train_x)}")
        logging.debug("Model sizes used: {}".format(sorted(train_x[:, 5].unique().cpu().numpy())))
        logging.debug("Steps used: {}".format(sorted(train_x[:, 6].unique().cpu().numpy())))
        logging.debug(f"Objective range: [{train_obj.min().item():.4f}, {train_obj.max().item():.4f}]")
        
        return mll, model

    def optimize_acquisition_function(self, model, best_f, alpha=1.0):
        """Optimize the cost-aware EI acquisition function."""
        acq_func = CostAwareEI(model=model, best_f=best_f, alpha=alpha)
        
        # Log model predictions at different fidelities
        with torch.no_grad():
            test_points = []
            # Sample points at different model sizes
            for size in [20, 300, 1000]:
                for step in [100, 10000, 19600]:
                    weights = torch.zeros(5, **tkwargs) # Use zero weights for testing
                    point = torch.cat([weights, torch.tensor([size, step], **tkwargs)])
                    test_points.append(point)
            test_points = torch.stack(test_points)
            
            posterior = model.posterior(test_points)
            means = posterior.mean.cpu().numpy()
            variances = posterior.variance.cpu().numpy()
            
            logging.debug("\nModel predictions at test points:")
            for i, (size, step) in enumerate([(s, st) for s in [20, 300, 1000] for st in [100, 10000, 19600]]):
                logging.debug(f"Model size: {size}M, Steps: {step}")
                logging.debug(f"  Mean: {means[i, 0]:.4f}")
                logging.debug(f"  Std: {np.sqrt(variances[i, 0]):.4f}")
        
        # Continue with existing optimization code...
        n_restarts = 5 if not SMOKE_TEST else 2
        raw_samples = 512 if not SMOKE_TEST else 64

        logging.debug("Starting optimization process...")

        # Generate initial points with valid discrete values [raw_samples, 7]
        X_init = generate_random_valid_points(raw_samples, **tkwargs)
        logging.debug(f"X_init shape after generation: {X_init.shape}")

        # Add q dimension for acquisition function [raw_samples, 1, 7]
        X_init_q = X_init.unsqueeze(1)
        logging.debug(f"X_init_q shape after unsqueeze: {X_init_q.shape}")

        # Evaluate acquisition function at initial points
        with torch.no_grad():
            logging.debug("Calling acquisition function...")
            acq_values = acq_func(X_init_q)
            logging.debug(f"acq_values shape: {acq_values.shape}")

        # Select top points as initial conditions [n_restarts, 7]
        top_indices = torch.topk(acq_values.squeeze(), n_restarts).indices
        logging.debug(f"top_indices shape: {top_indices.shape}")
        initial_conditions = X_init[top_indices]
        logging.debug(
            f"initial_conditions shape before unsqueeze: {initial_conditions.shape}"
        )

        # Add q dimension for optimize_acqf [n_restarts, 1, 7]
        initial_conditions = initial_conditions.unsqueeze(1)
        logging.debug(
            f"initial_conditions shape after unsqueeze: {initial_conditions.shape}"
        )

        # Optimize acquisition function
        logging.debug("Calling optimize_acqf...")
        candidate, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,
            num_restarts=n_restarts,
            raw_samples=raw_samples,
            options={
                "batch_limit": 1,
                "maxiter": 200,
                "nonnegative": False,
                "method": "L-BFGS-B",
            },
            return_best_only=True,
            batch_initial_conditions=initial_conditions,
        )
        logging.debug(f"candidate shape: {candidate.shape}")

        return candidate, acq_value

    def get_recommendation(self, model):
        """Get current recommendation at target settings (1B model, 19600 steps)."""
        logging.debug("\nStarting get_recommendation...")
        
        # Create acquisition function for posterior mean with fixed features
        pm = PosteriorMean(model)
        pm_fixed = FixedFeatureAcquisitionFunction(
            acq_function=pm,
            d=7,  # Total dimension
            columns=[5, 6],  # Fix model size and steps
            values=[self.reference_size, self.reference_steps],  # Fix to reference settings
        )
        
        # Optimize only the mixture weights (first 5 dimensions)
        bounds_weights = self.bounds[:, :5]
        logging.debug(f"Bounds shape: {self.bounds.shape}")
        logging.debug(f"Bounds_weights shape: {bounds_weights.shape}")
        
        # Generate and optimize initial conditions for weights only
        n_restarts = 20 if not SMOKE_TEST else 2
        raw_samples = 512 if not SMOKE_TEST else 64
        
        X_init = (
            torch.rand(raw_samples, 5, **tkwargs)
            * (bounds_weights[1] - bounds_weights[0])
            + bounds_weights[0]
        )
        logging.debug(f"X_init shape: {X_init.shape}")
        
        with torch.no_grad():
            # Evaluate posterior mean at initial points
            logging.debug("Evaluating posterior mean...")
            # Add q dimension for acquisition function
            X_init_q = X_init.unsqueeze(1)  # Shape: [raw_samples, 1, 5]
            logging.debug(f"X_init_q shape: {X_init_q.shape}")
            pm_values = pm_fixed(X_init_q)
            logging.debug(f"pm_values shape: {pm_values.shape}")
        
        # Select top points as initial conditions
        logging.debug("Selecting top points...")
        top_indices = torch.topk(pm_values.squeeze(), n_restarts).indices
        logging.debug(f"top_indices shape: {top_indices.shape}")
        
        initial_conditions = X_init[top_indices].unsqueeze(1)  # Shape: [n_restarts, 1, 5]
        logging.debug(f"initial_conditions shape: {initial_conditions.shape}")
        
        logging.debug("Starting optimize_acqf...")
        final_rec, _ = optimize_acqf(
            acq_function=pm_fixed,
            bounds=bounds_weights,
            q=1,
            num_restarts=n_restarts,
            raw_samples=raw_samples,
            options={"batch_limit": 5, "maxiter": 200},
            batch_initial_conditions=initial_conditions,
        )
        logging.debug(f"final_rec shape after optimize_acqf: {final_rec.shape}")
        
        # Construct full point with target fidelities
        final_rec_full = pm_fixed._construct_X_full(final_rec)
        logging.debug(f"final_rec_full shape: {final_rec_full.shape}")

        # Evaluate at reference settings
        raw_value = self.problem._raw_func(
            self.reference_size,
            self.reference_steps / 100,
            final_rec[0, :5].tolist(),  # weights - no q dimension needed
        )

        proportions = torch.exp(final_rec[0, :5]) / torch.sum(
            torch.exp(final_rec[0, :5])
        )
        logging.debug(
            "\nCurrent recommendation at reference settings (1B model, 19600 steps):"
        )
        logging.debug(f"Value: {raw_value:.4f}")
        logging.debug("Data mixture proportions:")
        for i, name in FEATURE_NAMES.items():
            logging.debug(f"{name}: {proportions[i].item():.3f}")

        # Log to wandb
        wandb.log(
            {
                "recommended_value": raw_value,
                "recommended_proportions": {
                    name: proportions[i].item() for i, name in FEATURE_NAMES.items()
                },
            }
        )

        return final_rec_full, raw_value

    def get_next_point(self, train_x, train_obj, alpha=1.0):
        """
        Get the next point to evaluate using cost-aware EI.

        Args:
            train_x: Training inputs
            train_obj: Training objectives
            alpha: Cost exponent (default=1.0)

        Returns:
            Tensor: Next point to evaluate
        """
        # Initialize and fit model
        mll, model = self.initialize_model(train_x, train_obj)
        fit_gpytorch_mll(mll)

        # Get best observed value
        best_f = train_obj.max()

        # Optimize acquisition function
        new_x, acq_value = self.optimize_acquisition_function(model, best_f, alpha)

        return new_x, acq_value

    def optimize(self, n_init=16, n_iterations=50, alpha=1.0):
        """Run the optimization loop."""
        logging.debug(f"Starting optimization with {n_iterations} iterations...")
        logging.debug(
            f"Target reference value: {self.reference_value:.4f} (1B model, {self.reference_steps} steps)"
        )

        # Generate initial data
        train_x, train_obj = self.generate_initial_data(n=n_init)

        # Compute initial costs and update history
        for i in range(len(train_x)):
            cost = compute_cost(train_x[i].unsqueeze(0))
            self.update_history(
                train_x[i].unsqueeze(0), train_obj[i].unsqueeze(0), cost
            )

        try:
            for i in range(n_iterations):
                # Initialize and fit model
                mll, model = self.initialize_model(train_x, train_obj)
                fit_gpytorch_mll(mll)

                # Get current recommendation
                rec, rec_value = self.get_recommendation(model)
                self.history["recommended_values"].append(rec_value)
                self.history["recommended_points"].append(rec)

                # Get next point
                new_x, acq_value = self.get_next_point(
                    self.history["train_x"], self.history["train_obj"], alpha=alpha
                )

                # Evaluate the point using the data model and negate for maximization
                new_obj = torch.tensor(
                    [
                        -self.problem._raw_func(  # Negate for maximization
                            new_x[0, 5].item(),  # model size
                            new_x[0, 6].item() / 100,  # step
                            new_x[0, :5].tolist(),  # weights
                        )
                    ],
                    **tkwargs,
                ).reshape(1, 1)

                # Compute cost
                new_cost = compute_cost(new_x)

                # Update history
                self.update_history(new_x, new_obj, new_cost)

                # Log progress (convert back to original scale for logging)
                logging.debug(f"\nIteration {i + 1}/{n_iterations}")
                logging.debug(f"Current value: {-new_obj.item():.4f}")  # Original scale
                logging.debug(
                    f"Best value: {self.history['best_value']:.4f}"
                )  # Original scale
                logging.debug(
                    f"Current recommendation value: {rec_value:.4f}"
                )  # At reference settings
                logging.debug(f"Current cost: {new_cost.item():.2e}")
                logging.debug(
                    f"Cumulative cost: {self.history['cumulative_costs'][-1]:.2e}"
                )
                logging.debug(f"Model size: {new_x[0, 5].item():.0f}M")
                logging.debug(f"Step: {new_x[0, 6].item():.0f}")

                # Log current proportions
                proportions = torch.exp(new_x[0, :5]) / torch.sum(
                    torch.exp(new_x[0, :5])
                )
                logging.debug("\nData mixture proportions:")
                for i, name in FEATURE_NAMES.items():
                    logging.debug(f"{name}: {proportions[i].item():.3f}")

        except KeyboardInterrupt:
            logging.warning("\nOptimization interrupted by user")

        return self.history


def main():
    # Initialize wandb and set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"eimf_test_{timestamp}"
    log_dir = setup_logging(run_name)

    wandb.init(
        project="botorch-mf",
        name=run_name,
        dir=str(log_dir),  # Store wandb files in the run directory
        config={
            "timestamp": timestamp,
            "smoke_test": SMOKE_TEST,
            "valid_model_sizes": VALID_MODEL_SIZES,
            "step_range": [min(VALID_STEPS), max(VALID_STEPS)],
            "step_multiple": 100,
            "alpha": 0.1,
            "n_init": 2 if SMOKE_TEST else 4,
            "n_iterations": 2 if SMOKE_TEST else 20,
            "metric": METRIC_NAMES[4],
            "reference_model_size": 1000,  # 1B parameters
            "reference_steps": 19600,
        },
    )

    try:
        logging.info("Initializing optimizer with Stack Exchange Cross Entropy metric")
        optimizer = EIMultiFidelityOptimizer(metric_index=4)

        # Run optimization
        n_init = 5 if SMOKE_TEST else 16
        n_iterations = 10 if SMOKE_TEST else 50
        history = optimizer.optimize(n_init=n_init, n_iterations=n_iterations)

        # Log final results
        logging.info("\nOptimization completed!")
        logging.info(f"Reference value (1B, 19600 steps): {optimizer.reference_value:.4f}")
        logging.info(f"Best value found: {history['best_value']:.4f}")
        logging.info("Best point:")
        logging.info(f"  Weights: {history['best_point'][0, :5]}")
        proportions = torch.exp(history["best_point"][0, :5]) / torch.sum(
            torch.exp(history["best_point"][0, :5])
        )
        logging.info("\nBest data mixture proportions:")
        for i, name in FEATURE_NAMES.items():
            logging.info(f"{name}: {proportions[i].item():.3f}")
        logging.info(f"  Model size: {history['best_point'][0, 5]:.0f}M")
        logging.info(f"  Step: {history['best_point'][0, 6]:.0f}")
        logging.info(f"  Cost: {history['best_cost']:.2e}")
        logging.info(f"Total cost: {history['cumulative_costs'][-1]:.2e}")

        # Create and save optimization plots
        plt.figure(figsize=(12, 8))

        # Plot best value vs iteration
        plt.subplot(2, 2, 1)
        plt.plot(history["best_values"], label="Best Observed")
        plt.plot(history["recommended_values"], label="Recommended at Target")
        plt.axhline(y=optimizer.reference_value, color="r", linestyle="--", label="Reference")
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.title("Optimization Progress")
        plt.legend()

        # Plot cumulative cost vs iteration
        plt.subplot(2, 2, 2)
        plt.plot(history["cumulative_costs"])
        plt.xlabel("Iteration")
        plt.ylabel("Cumulative Cost")
        plt.title("Cumulative Cost")

        plt.tight_layout()

        # Save plot
        plot_path = log_dir / "optimization_progress.png"
        plt.savefig(plot_path)
        wandb.log({"optimization_progress": wandb.Image(str(plot_path))})

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        logging.error(traceback.format_exc())

    # Close wandb run
    wandb.finish()


# Constants from train_eikg.py
METRIC_NAMES = {
    0: "Train Cross Entropy",
    1: "Common Crawl Cross Entropy",
    2: "C4 Cross Entropy",
    3: "Wikipedia Cross Entropy",
    4: "Stack Exchange Cross Entropy",  # Current default
    5: "Github Cross Entropy",
    6: "ArXiv Cross Entropy",
    7: "Book Cross Entropy",
    8: "Hellaswag Accuracy",
    9: "PIQA Accuracy",
    10: "ARC Easy Accuracy",
}

FEATURE_NAMES = {
    0: "RedPajamaWikipedia",
    1: "RedPajamaStackExchange",
    2: "RedPajamaGithub",
    3: "RedPajamaArXiv",
    4: "RedPajamaBook",
}


class DataMixtureMultiFidelityKernel(ProductKernel):
    """Custom kernel for multi-fidelity optimization of data mixture proportions.

    Combines:
    - Matern kernel for mixture weights (dims 0-4)
    - ExponentialDecay for model size (dim 5)
    - ExponentialDecay for steps (dim 6)
    """

    def __init__(
        self,
        batch_shape=torch.Size([]),
        nu=2.5,
    ):
        super().__init__()

        # Kernel for data mixture weights
        self.mixture_kernel = MaternKernel(
            nu=nu,
            ard_num_dims=5,  # One lengthscale per weight dimension
            batch_shape=batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
            active_dims=list(range(5)),  # First 5 dimensions are weights
        )

        # Kernel for model size fidelity
        self.size_kernel = ExponentialDecayKernel(
            batch_shape=batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
            offset_prior=GammaPrior(3.0, 6.0),
            power_prior=GammaPrior(3.0, 6.0),
            active_dims=[5],  # Dimension 5 is model size
        )

        # Kernel for training steps fidelity
        self.step_kernel = ExponentialDecayKernel(
            batch_shape=batch_shape,
            lengthscale_prior=GammaPrior(3.0, 6.0),
            offset_prior=GammaPrior(3.0, 6.0),
            power_prior=GammaPrior(3.0, 6.0),
            active_dims=[6],  # Dimension 6 is training steps
        )

        # Register all kernels
        self.kernels = ModuleList(
            [
                self.mixture_kernel,
                self.size_kernel,
                self.step_kernel,
            ]
        )


class DataMixtureMultiFidelityGP(SingleTaskGP):
    """Custom GP for multi-fidelity optimization of data mixture proportions."""

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        train_Yvar: Optional[Tensor] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
    ) -> None:
        if outcome_transform is None:
            outcome_transform = Standardize(m=1)

        # Initialize the kernel
        covar_module = ScaleKernel(
            DataMixtureMultiFidelityKernel(),
            batch_shape=train_X.shape[:-2],
            outputscale_prior=GammaPrior(2.0, 0.15),
        )

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            train_Yvar=train_Yvar,
            covar_module=covar_module,
            outcome_transform=outcome_transform,
        )

    def forward(self, x):
        logging.debug(f"\nGP forward pass with input shape: {x.shape}")
        if hasattr(self, 'train_inputs'):
            logging.debug(f"Train inputs shape: {self.train_inputs[0].shape}")
        mean_x = self.mean_module(x)
        logging.debug(f"Mean shape: {mean_x.shape}")
        covar_x = self.covar_module(x)
        logging.debug(f"Covariance shape: {covar_x.shape}")
        return MultivariateNormal(mean_x, covar_x)

    def project_to_target(self, X: Tensor) -> Tensor:
        """Project X to target fidelities (1B model, 19600 steps)."""
        if not isinstance(X, Tensor):
            X = torch.tensor(X, dtype=torch.float64)

        X_target = X.clone()
        X_target[..., 5] = 1000.0  # Set model size to 1B
        X_target[..., 6] = 19600.0  # Set steps to max

        return X_target


if __name__ == "__main__":
    main()
