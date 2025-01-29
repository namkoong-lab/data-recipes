import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
import wandb

# Add the root directory to the path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

import torch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_mll
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean, qExpectedImprovement
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.test_functions.multi_fidelity import AugmentedHartmann
from opt_algos.benchmarks import DataModelBenchmark
from typing import Tuple, Dict, Optional, Literal
from dataclasses import dataclass
import enum


# Global settings
GPU_ID = 2  # Change this to use different GPU
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

tkwargs = {
    "dtype": torch.double,
    "device": torch.device(DEVICE),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")

# Constants
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

NUM_RESTARTS = 5 if not SMOKE_TEST else 2
RAW_SAMPLES = 128 if not SMOKE_TEST else 4
BATCH_SIZE = 4

class AcquisitionType(enum.Enum):
    EI = "ei"
    KG = "kg"

@dataclass
class OptimizationConfig:
    metric_index: int = 4
    fidelities: torch.Tensor = torch.tensor([100.0, 150.0, 196.0], **tkwargs)
    bounds: torch.Tensor = torch.tensor([
        [-3.0, -3.0, -3.0, -3.0, -3.0, 1.0],  # lower bounds
        [3.0, 3.0, 3.0, 3.0, 3.0, 196.0],  # upper bounds
    ], **tkwargs)
    target_fidelities: Dict[int, float] = None
    cost_model: Optional[AffineFidelityCostModel] = None
    
    def __post_init__(self):
        if self.target_fidelities is None:
            self.target_fidelities = {5: 196.0}
        if self.cost_model is None:
            self.cost_model = AffineFidelityCostModel(fidelity_weights={5: 1.0}, fixed_cost=1000.0)

class BayesianOptimizer:
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.problem = DataModelBenchmark(metric_index=config.metric_index, device=DEVICE)
        print(f"Instantiating benchmark with y=metric {METRIC_NAMES[config.metric_index]}")
        
        # Create utility function that considers both information gain and cost
        self.cost_aware_utility = InverseCostWeightedUtility(cost_model=config.cost_model)
        
        # Initialize history tracking
        self.history = {
            "costs": [],
            "best_values": [],  # Best observed values
            "recommended_values": [],  # Best values from get_recommendation
            "acq_type": None
        }
        
        # Initialize wandb run if not already initialized
        if wandb.run is None:
            wandb.init(
                project="botorch",
                config={
                    "metric": METRIC_NAMES[config.metric_index],
                    "fidelities": config.fidelities.tolist(),
                    "bounds": config.bounds.tolist(),
                    "device": DEVICE,
                }
            )

    def generate_initial_data(self, n=16):
        train_x = torch.rand(n, 5, **tkwargs) * 6.0 - 3.0  # Scale to [-3.0, 3.0]
        train_f = self.config.fidelities[torch.randint(3, (n, 1))].to(**tkwargs)  # Ensure fidelities are on correct device
        train_x_full = torch.cat((train_x, train_f), dim=1)
        train_obj = -torch.tensor(
            [
                self.problem.func(int(z.item()), x.tolist())[-1][1]
                for x, z in zip(train_x, train_f)
            ],
            **tkwargs,
        ).unsqueeze(-1)
        return train_x_full, train_obj

    def initialize_model(self, train_x, train_obj):
        model = SingleTaskMultiFidelityGP(
            train_x, train_obj, outcome_transform=Standardize(m=1), data_fidelities=[5]
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def get_acquisition_function(self, model: SingleTaskMultiFidelityGP, acq_type: AcquisitionType):
        if acq_type == AcquisitionType.EI:
            return self._get_ei(model)
        elif acq_type == AcquisitionType.KG:
            return self._get_mfkg(model)
        else:
            raise ValueError(f"Unknown acquisition type: {acq_type}")

    def _get_ei(self, model):
        return FixedFeatureAcquisitionFunction(
            acq_function=qExpectedImprovement(model=model, best_f=model.train_targets.max()),
            d=6,  # Total dimensions including fidelity
            columns=[5],  # Fix fidelity dimension
            values=[196.0],  # Fix fidelity to highest value
        )

    def _get_mfkg(self, model):
        # First get current best value at highest fidelity
        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=6,
            columns=[5],
            values=[196.0],
        )

        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=self.config.bounds[:, :-1],
            q=1,
            num_restarts=10 if not SMOKE_TEST else 2,
            raw_samples=1024 if not SMOKE_TEST else 4,
            options={"batch_limit": 10, "maxiter": 200},
        )

        return qMultiFidelityKnowledgeGradient(
            model=model,
            num_fantasies=128 if not SMOKE_TEST else 2,
            current_value=current_value,
            cost_aware_utility=self.cost_aware_utility,
            project=self._project_to_target,
        )

    def _project_to_target(self, X):
        return project_to_target_fidelity(X=X, target_fidelities=self.config.target_fidelities)

    def optimize_and_get_observation(self, acq_func, acq_type: AcquisitionType):
        """Optimizes acquisition function and returns new candidates, observations, and cost."""
        if acq_type == AcquisitionType.EI:
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=self.config.bounds[:, :-1],
                q=BATCH_SIZE,
                num_restarts=10,
                raw_samples=512,
                options={"batch_limit": 5, "maxiter": 200},
            )
            candidates = acq_func._construct_X_full(candidates)
        else:  # KG
            candidates, _ = optimize_acqf_mixed(
                acq_function=acq_func,
                bounds=self.config.bounds,
                fixed_features_list=[{5: 100.0}, {5: 150.0}, {5: 196.0}],
                q=BATCH_SIZE,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                options={"batch_limit": 5, "maxiter": 200},
            )

        cost = self.config.cost_model(candidates).sum()
        new_x = candidates.detach()
        raw_obj = torch.tensor(
            [
                self.problem.func(int(z.item()), x.tolist())[-1][1]
                for x, z in zip(new_x[:, :-1], new_x[:, -1])
            ],
            **tkwargs,
        ).unsqueeze(-1)
        new_obj = -raw_obj
        
        # Update history
        self.history["costs"].append(cost.item())
        current_best = max(new_obj.max().item(), self.history["best_values"][-1] if self.history["best_values"] else float("-inf"))
        self.history["best_values"].append(current_best)
        
        # Log to wandb
        wandb.log({
            f"{acq_type.value}/cost": cost.item(),
            f"{acq_type.value}/best_value": current_best,
            f"{acq_type.value}/batch_size": len(new_x),
            f"{acq_type.value}/cumulative_cost": sum(self.history["costs"]),
        })
        
        print(f"candidates:\n{new_x}\n")
        print(f"observations:\n{new_obj}\n\n")
        return new_x, new_obj, cost

    def get_recommendation(self, model):
        rec_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=6,
            columns=[5],
            values=[196.0],
        )

        final_rec, _ = optimize_acqf(
            acq_function=rec_acqf,
            bounds=self.config.bounds[:, :-1],
            q=1,
            num_restarts=100,
            raw_samples=1024,
            options={"batch_limit": 5, "maxiter": 200},
        )

        final_rec = rec_acqf._construct_X_full(final_rec)

        raw_value = self.problem.func(int(final_rec[0, -1].item()), final_rec[0, :-1].tolist())[-1][1]
        print(f"recommended point:\n{final_rec}\n\ncross entropy value:\n{raw_value}")
        print(f"optimized objective value (negated cross entropy):\n{-raw_value}")

        proportions = torch.exp(final_rec[0, :-1]) / torch.sum(torch.exp(final_rec[0, :-1]))
        print("\nData mixture proportions:")
        for i, name in FEATURE_NAMES.items():
            print(f"{name}: {proportions[i]:.3f}")

        # Log recommendation to wandb
        wandb.log({
            f"{self.history['acq_type']}/recommended_value": -raw_value,
            f"{self.history['acq_type']}/proportions": {
                name: proportions[i].item() for i, name in FEATURE_NAMES.items()
            }
        })

        return final_rec

def convert_to_serializable(obj):
    """Convert numpy arrays and torch tensors to lists for JSON serialization."""
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def main():
    config = OptimizationConfig()
    
    # Create results directory with datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', f'run_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    print(f"\nResults will be saved in: {results_dir}")
    
    # Initialize wandb
    wandb.init(
        project="botorch",
        name=f"single_run_{timestamp}",
        config={
            "timestamp": timestamp,
            "n_iterations": N_ITER,
            "batch_size": BATCH_SIZE,
            "smoke_test": SMOKE_TEST,
        }
    )
    
    # Store results for plotting
    results = {}
    
    # Run optimization with both acquisition functions
    N_ITER = 3 if not SMOKE_TEST else 1
    
    try:
        # Generate initial data once
        initial_optimizer = BayesianOptimizer(config)
        initial_train_x, initial_train_obj = initial_optimizer.generate_initial_data(n=16)
        initial_best_value = initial_train_obj.max().item()
        
        print(f"Initial data generated with {len(initial_train_x)} points")
        print(f"Initial best value: {initial_best_value}")
        
        # Store initial data
        initial_data = {
            "initial_best_value": initial_best_value,
            "initial_points": len(initial_train_x),
            "timestamp": timestamp,
            "n_iterations": N_ITER,
            "batch_size": BATCH_SIZE,
        }
        
        for acq_type in [AcquisitionType.EI, AcquisitionType.KG]:
            print(f"\nRunning optimization with {acq_type.value} acquisition function")
            
            optimizer = BayesianOptimizer(config)
            optimizer.history["acq_type"] = acq_type.value
            
            # Use the same initial data
            train_x = initial_train_x.clone()
            train_obj = initial_train_obj.clone()
            
            # Initialize history with initial best value
            optimizer.history["costs"].append(0.0)
            optimizer.history["best_values"].append(initial_best_value)
            
            cumulative_cost = 0.0
            iteration_results = []
            
            try:
                for iteration in range(N_ITER):
                    print(f"\nIteration {iteration + 1}/{N_ITER}")
                    print(f"Current training data size: {len(train_x)}")
                    
                    mll, model = optimizer.initialize_model(train_x, train_obj)
                    fit_gpytorch_mll(mll)
                    
                    # Get current recommendation before acquisition
                    rec = optimizer.get_recommendation(model)
                    raw_value = optimizer.problem.func(int(rec[0, -1].item()), rec[0, :-1].tolist())[-1][1]
                    recommended_value = -raw_value
                    optimizer.history["recommended_values"].append(recommended_value)
                    print(f"Current recommended value: {recommended_value}")
                    
                    acq_func = optimizer.get_acquisition_function(model, acq_type)
                    new_x, new_obj, cost = optimizer.optimize_and_get_observation(acq_func, acq_type)
                    
                    # Check for NaN values
                    if torch.isnan(new_obj).any():
                        print("Warning: NaN values detected in observations")
                        continue
                        
                    train_x = torch.cat([train_x, new_x])
                    train_obj = torch.cat([train_obj, new_obj])
                    cumulative_cost += cost
                    
                    current_best = train_obj.max().item()
                    print(f"Current best value: {current_best}")
                    print(f"Cumulative cost: {cumulative_cost}")
                    
                    # Store iteration results
                    iteration_results.append({
                        "iteration": iteration + 1,
                        "best_value": current_best,
                        "cumulative_cost": cumulative_cost.item(),
                        "training_points": len(train_x)
                    })
                
            except Exception as e:
                print(f"Error during optimization with {acq_type.value}: {str(e)}")
                continue

            print(f"\nFinal recommendation for {acq_type.value}:")
            try:
                final_rec = optimizer.get_recommendation(model)
                print(f"Total cost: {cumulative_cost}\n")
                
                # Store final recommendation
                iteration_results.append({
                    "final_recommendation": convert_to_serializable(final_rec),
                    "total_cost": cumulative_cost.item()
                })
                
            except Exception as e:
                print(f"Error getting final recommendation: {str(e)}")
            print("--------------------------------")
            
            # Store results for plotting
            results[acq_type.value] = {
                "costs": convert_to_serializable(np.cumsum(optimizer.history["costs"])),
                "best_values": convert_to_serializable(optimizer.history["best_values"]),
                "recommended_values": convert_to_serializable(optimizer.history["recommended_values"]),
                "iterations": iteration_results
            }
        
        if results:
            # Save results to JSON
            results_data = {
                "metadata": initial_data,
                "optimization_results": convert_to_serializable(results)
            }
            
            with open(os.path.join(results_dir, 'results.json'), 'w') as f:
                json.dump(results_data, f, indent=2)
            
            # Create comparison plot
            plt.figure(figsize=(10, 6))
            for acq_type, data in results.items():
                plt.plot(data["costs"], data["best_values"], marker='o', linestyle='--', label=f"{acq_type.upper()} (Observed)")
                plt.plot(data["costs"], data["recommended_values"], marker='s', label=f"{acq_type.upper()} (Recommended)")
            
            plt.xlabel("Cumulative Cost")
            plt.ylabel("Best Objective Value")
            plt.title("Comparison of Acquisition Functions\n(Observed vs Recommended Values)")
            plt.legend()
            plt.grid(True)
            
            # Save plot to results directory
            plot_path = os.path.join(results_dir, 'acquisition_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"\nResults and plot saved in: {results_dir}")
        else:
            print("\nNo results to plot")
            
    except Exception as e:
        print(f"Fatal error in main: {str(e)}")
        import traceback
        traceback.print_exc()

    # Close wandb run at the end
    wandb.finish()

if __name__ == "__main__":
    main()
