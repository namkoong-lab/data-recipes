import json
import warnings

from linear_operator.utils.warnings import NumericalWarning

warnings.filterwarnings("ignore", category=NumericalWarning)

import argparse
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

FLOPS = {
    100: 1,
    2: 2090524455 / 161264981936,
    6: 5211827866 / 161264981936,
    15: 12069704997 / 161264981936,
    30: 23823782173 / 161264981936,
    50: 34933622501 / 161264981936,
    70: 48105020743 / 161264981936,
}


def get_best_y_at_scale(func, x, args):
    x = x.numpy()
    if len(x.shape) == 1:
        x = x.reshape(1, -1)

    y = func(197, 100, x[:, :5])
    return y


def main(args, folder, logger):
    data_benchmark = benchmarks.DataModelBenchmark(metric_index=args.metric_index)
    if args.metric_index >= 8:
        # Accuracy metric. We want to maximize it.
        func = lambda z, m, x: -data_benchmark._raw_func_with_model_scale(
            z, m, x, with_exp=False
        )
    else:
        func = lambda z, m, x: data_benchmark._raw_func_with_model_scale(
            z, m, x, with_exp=False
        )  # (z, m, x)

    SCALES = np.array(args.scales) / args.scale_scale
    TIMESTEPS = np.array(args.timesteps) / args.step_scale

    def get_random_points(k, return_x=False):
        rd_prop = np.random.dirichlet(np.ones(5), size=k)

        # Scale range to be from 0 to 1
        rd_scale = np.random.choice(args.ini_scales, size=k) / args.scale_scale
        rd_timestep = np.random.choice(args.ini_timesteps, size=k) / args.step_scale

        if return_x:
            rd_x = np.concatenate(
                [rd_prop, rd_scale[:, None], rd_timestep[:, None]], axis=1
            )
            return rd_x
        else:
            return rd_prop, rd_scale, rd_timestep

    # sample k random points from scale 2 and scale 15
    rd_prop, rd_scale, rd_timestep = get_random_points(args.ini_revealed)

    train_x, train_y = [], []
    for i in range(args.ini_revealed):
        step = np.round(rd_timestep[i] * args.step_scale).astype(int)
        for s in range(1, step + 1):
            train_x.append(
                np.concatenate([rd_prop[i], [rd_scale[i], s / args.step_scale]])
            )
            train_y.append(func(s, rd_scale[i] * args.scale_scale, rd_prop[i]))
    train_x, train_y = torch.tensor(train_x), torch.tensor(train_y)

    logger.info(f"Number of training points: {len(train_x)}")
    logger.info(f"Initial y: {train_y}")

    # Set default tensor type to double
    torch.set_default_tensor_type(torch.DoubleTensor)

    # Train GP
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            # Choose either ConstantMean or LinearMean, but handle appropriately
            if args.mean_type == "constant":
                self.mean_module = gpytorch.means.ConstantMean()
            else:  # linear
                self.mean_module = gpytorch.means.LinearMean(input_size=7)

            if args.kernel_type == "rbf":
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel()
                )
            elif args.kernel_type == "decay":
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    bptu.DataMixtureMultiFidelityKernel()
                )
            elif args.kernel_type == "rbf_product":
                self.covar_module = (
                    gpytorch.kernels.RBFKernel(
                        active_dims=torch.tensor([0, 1, 2, 3, 4])
                    )
                    * gpytorch.kernels.RBFKernel(active_dims=torch.tensor([5]))
                    * gpytorch.kernels.RBFKernel(active_dims=torch.tensor([6]))
                )
            else:
                raise ValueError(f"Kernel type {args.kernel_type} not recognized.")

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
    for i in range(args.gp_training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        logger.info(f"Iter {i+1}/{args.gp_training_iter} - Loss: {loss.item()}")

    if args.kernel_type == "rbf":
        logger.info(
            f"Final lengthscale: {model.covar_module.base_kernel.lengthscale.item()}"
        )
    elif args.kernel_type == "rbf_product":
        logger.info(
            f"Final lengthscale: {model.covar_module.kernels[0].lengthscale.item()}"
        )
        logger.info(
            f"Final lengthscale: {model.covar_module.kernels[1].lengthscale.item()}"
        )
        logger.info(
            f"Final lengthscale: {model.covar_module.kernels[2].lengthscale.item()}"
        )
    # Log mean function parameters based on type
    if isinstance(model.mean_module, gpytorch.means.ConstantMean):
        logger.info(f"Final mean constant: {model.mean_module.constant.item()}")
        model.mean_module.constant.data.fill_(0.8)  # If you still want to override it
    else:  # LinearMean
        logger.info(f"Final mean weights: {model.mean_module.weights.detach().numpy()}")
        # Don't override linear weights as it would break the learned relationships

    # Set length scale
    if args.kernel_type == "rbf":
        model.covar_module.base_kernel.lengthscale = 1.5

    # Evaluate GP Bayesopt
    model.eval()
    likelihood.eval()

    ## Calculate initial cost
    ini_cost = 0
    for scale in SCALES:
        cc = (
            np.sum(rd_timestep[rd_scale == scale])
            * args.step_scale
            * FLOPS[int(scale * args.scale_scale)]
        )
        logger.info(f"Initial cost for scale {scale * args.scale_scale}: {cc}")
        ini_cost += cc
    cost = [ini_cost]

    ## Calculate initial best point
    best_y = train_y.min()
    best_idx = train_y.argmin()
    best_x = train_x[best_idx]
    logger.info(f"Best point so far: {best_y} at {best_x}")

    best_xs = [best_x]
    best_ys = [best_y]
    best_yass = [get_best_y_at_scale(func, best_x, args)]
    best_eis = []

    ## Run Bayesopt
    for i in tqdm(range(args.num_revealed), desc="Revealing labels"):
        # Find the point with the highest EI
        def ei_to_minimize(x, scale, timestep, model, likelihood, train_y):
            x = torch.tensor(x, dtype=torch.double).reshape(1, -1)
            p = torch.exp(x)
            x = p / torch.sum(p)
            x = torch.cat(
                [
                    x,
                    torch.tensor([scale, timestep], dtype=torch.double).reshape(1, -1),
                ],
                dim=1,
            )

            return (
                -bptu.compute_ei(model, likelihood, train_y, x).detach().numpy().item()
            )

        def ei_grad(x, scale, timestep, model, likelihood, train_y):
            x = torch.tensor(x, dtype=torch.double).reshape(1, -1)
            p = torch.exp(x)
            x = p / torch.sum(p)
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
                x0s = np.random.uniform(-3, 3, size=(args.num_search_per_fid, 5))

                results = list(
                    map(
                        lambda x0: minimize(
                            lambda x: ei_to_minimize(
                                x, scale, timestep, model, likelihood, train_y
                            ),
                            x0=x0[:5],
                            bounds=[(-3, 3)] * 5,
                            method=args.ei_optimizer,
                            jac=lambda x: ei_grad(
                                x, scale, timestep, model, likelihood, train_y
                            ),
                        ),
                        x0s,
                    )
                )

                best_result = max(results, key=lambda x: -x.fun)
                cur_x = np.array(best_result.x)
                cur_x = np.exp(cur_x) / np.sum(np.exp(cur_x))
                ei_results[scale][timestep] = (cur_x, -best_result.fun)

        # Scale EI by cost of the operation
        scaled_ei_results = defaultdict(defaultdict)
        max_scaled_ei = -1

        alpha_decay = args.alpha * args.alpha_decay**i
        logger.info(f"At iteration {i}, cost alpha: {alpha_decay}")
        for scale in ei_results.keys():
            for timestep in ei_results[scale].keys():
                run_cost = (
                    FLOPS[int(scale * args.scale_scale)] * timestep * args.step_scale
                )

                scaled_ei_results[scale][timestep] = (
                    ei_results[scale][timestep][0],
                    ei_results[scale][timestep][1] / (run_cost**alpha_decay),
                )
                scaled_ei = scaled_ei_results[scale][timestep][1]
                if scaled_ei > max_scaled_ei:
                    max_scaled_ei = scaled_ei

                    best_scale = scale
                    best_timestep = timestep
                    best_x = ei_results[scale][timestep][0]

        best_ei = ei_results[best_scale][best_timestep][1]
        best_eis.append(best_ei)
        best_ei_y = func(
            best_timestep * args.step_scale, best_scale * args.scale_scale, best_x
        )
        logger.info(
            f"\n\tBest ei: {best_ei},"
            f"\n\tBest scaled ei: {scaled_ei_results[best_scale][best_timestep][1]},"
            f"\n\tscale: {best_scale * args.scale_scale},"
            f"\n\ttimestep: {best_timestep * args.step_scale},"
            f"\n\tx: {best_x}"
            f"\n\ty: {best_ei_y}"
        )

        best_ei_at_scale = ei_results[100 / args.scale_scale][197 / args.step_scale][1]
        best_ei_at_scale_x = ei_results[100 / args.scale_scale][197 / args.step_scale][
            0
        ]
        logger.info(
            f"\n\tBest ei at full fidelity: {best_ei_at_scale}"
            f"\n\tBest scaled ei at full fidelity: {scaled_ei_results[100 / args.scale_scale][197 / args.step_scale][1]}"
            f"\n\tx: {best_ei_at_scale_x}"
            f"\n\ty: {func(197, 100, best_ei_at_scale_x)}"
        )

        # Update the model with the new point
        new_x = torch.tensor(
            np.concatenate(
                [
                    best_x,
                    np.array([best_scale, best_timestep]),
                ]
            ).reshape(1, -1)
        )
        new_y = torch.tensor([best_ei_y])

        train_x = torch.cat([train_x, new_x], dim=0)
        train_y = torch.cat([train_y, new_y], dim=0)

        model.set_train_data(train_x, train_y, strict=False)

        # Sanity check posterior
        with torch.no_grad():
            model.eval()
            likelihood.eval()
            m_out = model(new_x)
            observed_pred = likelihood(m_out)
            mean = observed_pred.mean
            covar = observed_pred.variance.sqrt()

            logger.info(
                f"Sanity check model update:"
                f"\n\tNew y: {new_y}"
                f"\n\tPosterior mean: {mean}"
                f"\n\tPosterior variance: {covar}"
            )

        # Sanity check GP over scale
        with torch.no_grad():
            num_pts = 100
            test_x = new_x.repeat(num_pts, 1)
            scales = torch.tensor(
                np.linspace(2 / args.scale_scale, 100 / args.scale_scale, num_pts)
            )
            test_x[:, -2] = scales

            model.eval()
            likelihood.eval()
            m_out = model(test_x)
            observed_pred = likelihood(m_out)
            mean = observed_pred.mean
            lower, upper = observed_pred.confidence_region()

            plt.plot(scales, mean, label="Mean")
            plt.fill_between(scales, lower, upper, alpha=0.1, label="Confidence")
            plt.scatter(best_scale, best_ei_y, color="red", label="Revealed point")
            plt.legend()
            plt.savefig(f"{folder}/gp_scale_{i}.png")
            plt.close()

        # Calculate cost
        cur_cost = (
            best_timestep * args.step_scale * FLOPS[int(best_scale * args.scale_scale)]
        )
        cost.append(cur_cost)

        # Report best point so far
        best_y = train_y.min()
        best_idx = train_y.argmin()
        best_x = train_x[best_idx]
        best_yas = get_best_y_at_scale(func, best_x, args)

        logger.info(f"Best point so far: {best_y} at {best_x}")
        logger.info(f"Best point so far at full fidelity: {best_yas}")
        logger.info(f"Current cost: {cur_cost}")
        logger.info(f"Cumuative cost: {sum(cost)}")

        best_xs.append(best_x)
        best_ys.append(best_y)
        best_yass.append(best_yas)

        # Save results
        np.save(f"{folder}/best_xs.npy", torch.stack(best_xs).numpy())
        np.save(f"{folder}/best_ys.npy", torch.stack(best_ys).numpy())
        np.save(f"{folder}/best_yass.npy", np.array(best_yass))
        np.save(f"{folder}/cost.npy", np.array(cost))
        np.save(f"{folder}/cumcost.npy", np.cumsum(np.array(cost)))
        np.save(f"{folder}/best_eis.npy", np.array(best_eis))

        # Update model length scale
        if best_ei < args.ei_threshold and args.kernel_type == "rbf":
            logger.info("EI too low.")
            new_scale = (
                model.covar_module.base_kernel.lengthscale * args.lengthscale_decay
            )
            logger.info(f"Updating lengthscale to {new_scale}")
            model.covar_module.base_kernel.lengthscale = new_scale

        if best_ei_at_scale < args.ei_threshold and args.kernel_type == "rbf_product":
            logger.info("EI too low at full fidelity.")
            new_scale = (
                model.covar_module.kernels[1].lengthscale * args.lengthscale_decay
            )
            logger.info(f"Updating model scale's lengthscale to {new_scale}")
            model.covar_module.kernels[1].lengthscale = new_scale

            new_scale = (
                model.covar_module.kernels[2].lengthscale * args.lengthscale_decay
            )
            logger.info(f"Updating timestep's lengthscale to {new_scale}")
            model.covar_module.kernels[2].lengthscale = new_scale


def get_args():
    parser = argparse.ArgumentParser(description="Run EI optimizer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--metric_index", type=int, default=4, help="Index of the metric to optimize"
    )
    parser.add_argument(
        "--kernel_type",
        type=str,
        default="rbf_product",
        help="The type of GP kernel to use",
    )
    parser.add_argument(
        "--num_revealed", type=int, default=100, help="Number of revealed labels"
    )
    parser.add_argument(
        "--ini_revealed",
        type=int,
        default=20,
        help="Number of initial revealed",
    )
    parser.add_argument(
        "--num_search_per_fid",
        type=int,
        default=5,
        help="Number of EI search per fidelity",
    )
    parser.add_argument(
        "--gp_training_iter",
        type=int,
        default=50,
        help="Number of GP training iterations",
    )
    parser.add_argument(
        "--scale_scale", type=int, default=100.0, help="Scale scale (for GP input)"
    )
    parser.add_argument(
        "--step_scale", type=int, default=197.0, help="Step scale (for GP input)"
    )
    parser.add_argument(
        "--scales",
        type=int,
        nargs="+",
        default=[2, 6, 15, 30, 50, 70, 100],
        help="Scales to search",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[60, 120, 197],
        help="Timesteps to search",
    )
    parser.add_argument(
        "--ini_scales",
        type=int,
        nargs="+",
        default=[2, 6, 15, 30, 50, 70, 100],
        help="Scales to sample initial points from",
    )
    parser.add_argument(
        "--ini_timesteps",
        type=int,
        nargs="+",
        default=np.arange(1, 10).tolist(),
        help="Timesteps to sample initial points from",
    )
    parser.add_argument(
        "--ei_optimizer",
        type=str,
        default="L-BFGS-B",
        help="Optimizer to use for EI optimization",
    )
    parser.add_argument(
        "--ei_threshold",
        type=float,
        default=1e-4,
        help="EI threshold to update lengthscale",
    )
    parser.add_argument(
        "--lengthscale_decay", type=float, default=0.95, help="Lengthscale decay factor"
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Power of cost")
    parser.add_argument(
        "--alpha_decay", type=float, default=0.99, help="Alpha decay factor"
    )
    parser.add_argument(
        "--mean_type", type=str, default="linear", help="Type of mean function"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    mscu.set_seed(args.seed)
    METRIC_NAMES = {
        0: "TrainCrossEntropy",
        1: "CommonCrawlCrossEntropy",
        2: "C4CrossEntropy",
        3: "WikipediaCrossEntropy",
        4: "StackExchangeCrossEntropy",
        5: "GithubCrossEntropy",
        6: "ArXivCrossEntropy",
        7: "BookCrossEntropy",
        8: "HellaswagAccuracy",
        9: "PIQAAccuracy",
        10: "ARCEasyAccuracy",
    }
    args.metric_name = METRIC_NAMES[args.metric_index]

    folder = (
        f"./results/ei_optimizer/metric={args.metric_name}"
        f"/kernel={args.kernel_type}_mean={args.mean_type}"
        f"/num_revealed={args.num_revealed}_ini_revealed={args.ini_revealed}"
        f"/scale_scale={args.scale_scale}_step_scale={args.step_scale}"
        f"/num_search_per_fid={args.num_search_per_fid}"
        f"/alpha={args.alpha}_alpha_decay={args.alpha_decay}"
        f"/seed={args.seed}"
    )
    mscu.make_folder(folder)

    fname = f"{folder}/ei_optimizer.log"
    logger = mscu.get_logger(filename=fname)

    logger.info(f"Running EI optimizer for metric {args.metric_name}.")
    logger.info(f"Arguments: {json.dumps(vars(args), indent=2)}")

    main(args, folder, logger)
