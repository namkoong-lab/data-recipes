import os
import sys

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
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
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
        self.problem = DataModelBenchmark(metric_index=config.metric_index)
        print(f"Instantiating benchmark with y=metric {METRIC_NAMES[config.metric_index]}")
        
        # Create utility function that considers both information gain and cost
        self.cost_aware_utility = InverseCostWeightedUtility(cost_model=config.cost_model)
        
    def generate_initial_data(self, n=16):
        train_x = torch.rand(n, 5, **tkwargs) * 6.0 - 3.0  # Scale to [-3.0, 3.0]
        train_f = self.config.fidelities[torch.randint(3, (n, 1))]
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

        return final_rec

def main():
    config = OptimizationConfig()
    optimizer = BayesianOptimizer(config)
    
    N_ITER = 3 if not SMOKE_TEST else 1
    cumulative_cost = 0.0

    # Run optimization with both acquisition functions
    # for acq_type in [AcquisitionType.EI, AcquisitionType.KG]:
    for acq_type in [AcquisitionType.KG]:
        print(f"\nRunning optimization with {acq_type.value} acquisition function")
        train_x, train_obj = optimizer.generate_initial_data(n=16)
        
        for _ in range(N_ITER):
            mll, model = optimizer.initialize_model(train_x, train_obj)
            fit_gpytorch_mll(mll)
            
            acq_func = optimizer.get_acquisition_function(model, acq_type)
            new_x, new_obj, cost = optimizer.optimize_and_get_observation(acq_func, acq_type)
            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])
            cumulative_cost += cost

        print(f"\nFinal recommendation for {acq_type.value}:")
        final_rec = optimizer.get_recommendation(model)
        print(f"Total cost: {cumulative_cost}\n")
        print("--------------------------------")

if __name__ == "__main__":
    main()
