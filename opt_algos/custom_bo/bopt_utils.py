import gpytorch
import numpy as np
import torch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_noiseless_likelihood(num_train):
    return gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=torch.ones(num_train) * 1
    )


def compute_ei(model, likelihood, train_y, test_x, with_grad=False, xi=0.0):
    """EI for minimization"""
    model.eval()
    likelihood.eval()

    if with_grad:
        m_out = model(test_x)
        observed_pred = likelihood(m_out)
    else:
        with torch.no_grad():
            m_out = model(test_x)
            observed_pred = likelihood(m_out)
    mean = observed_pred.mean

    # Compute EI
    y_best = train_y.min()
    delta = (y_best + xi) - mean
    sigma = observed_pred.variance.sqrt()
    t1 = delta * torch.distributions.Normal(0, 1).cdf(delta / sigma)
    t2 = sigma * torch.exp(torch.distributions.Normal(0, 1).log_prob(delta / sigma))
    ei = t1 + t2
    return ei


def compute_lcb(model, likelihood, test_x, with_grad=False, beta=1.0):
    """LCB for minimization"""
    model.eval()
    likelihood.eval()

    if with_grad:
        m_out = model(test_x)
        observed_pred = likelihood(m_out)
    else:
        with torch.no_grad():
            m_out = model(test_x)
            observed_pred = likelihood(m_out)
    mean = observed_pred.mean

    # Compute LCB
    sigma = observed_pred.variance.sqrt()
    lcb = mean - beta * sigma
    return lcb
