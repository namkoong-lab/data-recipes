import warnings

from linear_operator.utils.warnings import NumericalWarning

warnings.filterwarnings("ignore", category=NumericalWarning)

from collections import defaultdict

import benchmarks
import bopt_utils as bptu
import gpytorch
import matplotlib.pyplot as plt
import misc_utils as mscu
import numpy as np
import optimizers
import torch
from scipy.optimize import minimize
from tqdm import tqdm

data_benchmark = benchmarks.DataModelBenchmark(metric_index=4)
func = data_benchmark._raw_func_with_model_scale  # (z, m, x)

fname = "ei_optimizer.log"
logger = mscu.get_logger(filename=fname)

FLOPS = {
    100: 1,
    2: 2090524455 / 161264981936,
    6: 5211827866 / 161264981936,
    15: 12069704997 / 161264981936,
    30: 23823782173 / 161264981936,
    50: 34933622501 / 161264981936,
    70: 48105020743 / 161264981936,
}

# Experiment parameters
k = 10
num_revealed = 15
num_search_per_fid = 5

training_iter = 20  # GP training iterations

SCALE_SCALE = 100
STEP_SCALE = 197
SCALES = np.array([2, 6, 15, 30, 50, 70, 100]) / SCALE_SCALE
# TIMESTEPS = np.arange(1, 197) / STEP_SCALE
TIMESTEPS = np.array([60, 120, 197]) / STEP_SCALE


logger.info(f"Sampling {k} random points to initiate GP")


def get_random_points(k, return_x=False):
    rd_prop = np.random.dirichlet(np.ones(5), size=k)

    # Scale range to be from 0 to 1
    rd_scale = np.random.choice([2, 15, 30, 50, 70, 100], size=k) / SCALE_SCALE
    rd_timestep = np.random.choice(np.arange(1, 20), size=k) / STEP_SCALE

    if return_x:
        rd_x = np.concatenate(
            [rd_prop, rd_scale[:, None], rd_timestep[:, None]], axis=1
        )
        return rd_x
    else:
        return rd_prop, rd_scale, rd_timestep


# sample k random points from scale 2 and scale 15
rd_prop, rd_scale, rd_timestep = get_random_points(k)

train_x, train_y = [], []
for i in range(k):
    step = np.round(rd_timestep[i] * STEP_SCALE).astype(int)
    for s in range(1, step + 1):
        train_x.append(np.concatenate([rd_prop[i], [rd_scale[i], s]]))
        train_y.append(func(s, rd_scale[i] * SCALE_SCALE, rd_prop[i]))
train_x, train_y = torch.tensor(train_x), torch.tensor(train_y)

logger.info(f"Number of training points: {len(train_x)}")
logger.info(f"Initial y: {train_y}")


# Train GP
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


likelihood = bptu.get_noiseless_likelihood(len(train_x))
model = ExactGPModel(train_x, train_y, likelihood)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
losses = []
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    losses.append(loss.item())
    optimizer.step()
    logger.info(f"Iter {i+1}/{training_iter} - Loss: {loss.item()}")

logger.info(f"Final lengthscale: {model.covar_module.base_kernel.lengthscale.item()}")
logger.info(f"Final mean: {model.mean_module.constant.item()}")

# Evaluate GP Bayesopt
model.eval()
likelihood.eval()

## Calculate initial cost
c2 = np.sum(rd_timestep[rd_scale == 2 / SCALE_SCALE]) * STEP_SCALE * FLOPS[2]
c15 = np.sum(rd_timestep[rd_scale == 15 / SCALE_SCALE]) * STEP_SCALE * FLOPS[15]

logger.info(f"Initial cost for scale 2: {c2}")
logger.info(f"Initial cost for scale 15: {c15}")

cost = [c2 + c15]

## Calculate initial best point
best_y = train_y.min()
best_idx = train_y.argmin()
best_x = train_x[best_idx]
logger.info(f"Best point so far: {best_y} at {best_x}")

best_xs = [best_x]
best_ys = [best_y]
best_eis = []

## Run Bayesopt
for i in tqdm(range(num_revealed), desc="Revealing labels"):
    # Find the point with the highest EI
    def ei_to_minimize(x, scale, timestep, model, likelihood, train_y):
        x = torch.tensor(x, dtype=torch.double).reshape(1, -1)
        x = torch.cat(
            [
                x,
                torch.tensor([scale, timestep], dtype=torch.double).reshape(1, -1),
            ],
            dim=1,
        )

        return -bptu.compute_ei(model, likelihood, train_y, x).detach().numpy().item()

    def ei_grad(x, scale, timestep, model, likelihood, train_y):
        x = torch.tensor(x, dtype=torch.double).reshape(1, -1)
        x = torch.cat(
            [
                x,
                torch.tensor([scale, timestep], dtype=torch.double).reshape(1, -1),
            ],
            dim=1,
        )

        x.requires_grad = True
        ei = -bptu.compute_ei(model, likelihood, train_y, x, with_grad=True)
        ei.backward()
        grad = x.grad[0, :5].squeeze(0).numpy()

        return grad

    ei_results = defaultdict(defaultdict)
    for scale in tqdm(SCALES, desc="Scales", position=1, leave=False):
        for timestep in TIMESTEPS:
            x0s = get_random_points(num_search_per_fid, return_x=True)

            # Bound x to sum to 1
            def sum_constraint(x):
                return np.sum(x) - 1  # Will equal 0 when sum is 1

            constraints = {"type": "eq", "fun": sum_constraint}
            results = list(
                map(
                    lambda x0: minimize(
                        lambda x: ei_to_minimize(
                            x, scale, timestep, model, likelihood, train_y
                        ),
                        x0=x0[:5],
                        bounds=[(0, 1)] * 5,
                        method="L-BFGS-B",
                        jac=lambda x: ei_grad(
                            x, scale, timestep, model, likelihood, train_y
                        ),
                        constraints=constraints,
                    ),
                    x0s,
                )
            )

            # Filter results not successful and not satisfying constraints
            results = list(
                filter(
                    lambda res: res.success and np.isclose(sum_constraint(res.x), 0.0),
                    results,
                )
            )
            if len(results) == 0:
                logger.warning(
                    f"Scale {scale * SCALE_SCALE} timestep {timestep * STEP_SCALE} has no successful optimizing results."
                    f"List of sum constraints:\n\t{[sum_constraint(res.x) for res in results]}"
                )
                continue

            best_result = max(results, key=lambda x: -x.fun)
            cur_x = best_result.x
            ei_results[scale][timestep] = (cur_x, -best_result.fun)

    # Scale EI by cost of the operation
    scaled_ei_results = defaultdict(defaultdict)
    max_scaled_ei = -1

    for scale in ei_results.keys():
        for timestep in ei_results[scale].keys():
            scaled_ei_results[scale][timestep] = (
                ei_results[scale][timestep][0],
                ei_results[scale][timestep][1]
                / FLOPS[int(scale * SCALE_SCALE)]
                / (timestep * STEP_SCALE),
            )
            scaled_ei = scaled_ei_results[scale][timestep][1]
            if scaled_ei > max_scaled_ei:
                max_scaled_ei = scaled_ei

                best_scale = scale
                best_timestep = timestep
                best_x = ei_results[scale][timestep][0]

    logger.info(
        f"Best ei: {max_scaled_ei},"
        f"\n\tscale: {best_scale * SCALE_SCALE},"
        f"\n\ttimestep: {best_timestep * STEP_SCALE},\n\tx: {best_x}"
    )
    best_ei = ei_results[best_scale][best_timestep][1]
    best_eis.append(best_ei)

    # Update the model with the new point
    int_best_time_step = np.round(best_timestep * STEP_SCALE).astype(int)
    new_x = torch.tensor(
        [
            np.concatenate((best_x, np.array([best_scale, t])))
            for t in range(1, int_best_time_step + 1)
        ]
    )
    new_y = torch.tensor(
        [
            func(t, best_scale * SCALE_SCALE, best_x)
            for t in range(1, int_best_time_step + 1)
        ]
    )
    logger.info(f"New y: {new_y}")
    train_x = torch.cat([train_x, new_x], dim=0)
    train_y = torch.cat([train_y, new_y], dim=0)

    model.set_train_data(train_x, train_y, strict=False)

    # Calculate cost
    cur_cost = best_timestep * 197 * FLOPS[int(best_scale * SCALE_SCALE)]
    cost.append(cur_cost)

    # Report best point so far
    best_y = train_y.min()
    best_idx = train_y.argmin()
    best_x = train_x[best_idx]
    logger.info(f"Best point so far: {best_y} at {best_x}")
    logger.info(f"Current cost: {cur_cost}")
    logger.info(f"Cumuative cost: {sum(cost)}")

    best_xs.append(best_x)
    best_ys.append(best_y)

    # Save results
    np.save("best_xs.npy", torch.stack(best_xs).numpy())
    np.save("best_ys.npy", torch.stack(best_ys).numpy())
    np.save("cost.npy", np.array(cost))
    np.save("best_eis.npy", np.array(best_eis))

    # Update model length scale
    new_scale = model.covar_module.base_kernel.lengthscale * 0.95
    logger.info(f"Updating lengthscale to {new_scale}")
    model.covar_module.base_kernel.lengthscale = new_scale
