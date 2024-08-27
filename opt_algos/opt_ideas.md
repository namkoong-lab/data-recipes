# Optimization algorithm ideas

This document outlines ideas for using deep sequence model in Bayesian Optimization.

## Bayesian Optimization

In Bayesian Optimization, we want to maximize a function $f(x)$ that is very expensive to evaluate, over a domain $\mathcal{A}$. A typical example is hyperparameter tuning/auto-ML, where $x$ are the hyperparameters, and $f(x)$ is $-1 \times$ the loss.

In its most general form, the framework looks something like this:
  1. Place a prior on the function $f$ (for example, a Gaussian process prior)
  2. At every step in the algorithm, use all points $(x_1, f(x_1)), \cdots, (x_n, f(x_n))$ observed so far to find a posterior probability distribution on $f$ (for example, a Gaussian process with posterior mean and variance functions)
  3. Using this posterior, we can find the posterior distribution of $f(x^\prime)$ for every $x^\prime \in \mathcal{A}$. 
  4. Use this posterior distribution to find the point in $\mathcal{A}$ that would be most valuable to observe next (see below)
  5. Observe it and repeat.

Three comments:
  - The most common example (and, as far as I know, the only principled example) of a prior that can be placed on $f$ in step (1) is a Gaussian Process prior.
  
    In practice, people use all kinds of other "back-of-the-envelope" priors. For example, in step (3), instead of finding the distribution of every $f(x^\prime)$ using a GP, they will fit a random forest to all points observed so far, and use it to predict $f(x^\prime)$ for every $x^\prime \in \mathcal{A}$. To get a *distribution* (rather than just a point estimate) they use the different trees in the RF to get an estimate of variance. This isn't really rigorously putting a "prior" on $f$, but it plays the same role.

  - Step (4) does a lot of heavy lifting - in reality, what we have here is a dynamic program in which the choice made in every period (what point to observe) is the decision, and the value of the function at that point is the reward. It's hard to solve the full DP here, so people use all kinds of Heuristics. For example:
    * "Thompson Sampling" - generate one realization of $f$ from the posterior, and choose the maximum point in that realization as the next point to observe
    * "Expected Improvement" - Let $f^\star$ be the current best objective value function; pick our new point $x^\prime \in \mathcal{A}$ to maximize $\mathbb{E} \left[ (f(x^\prime) - f^\star)^+ \right]$, where the expctation is over the posterior

  - There is one final complication - in some cases, we get to choose how much to "invest" into an evaluation of $f$. We can either spend a long time to get a very accurate (high fidelity) value of $f(x)$, or spend less time to get a less accurate (low fidelity) value. In the hyperparameter tuning example, a low fidelity estimate of $f(x)$ might be obtained by only training for very few epochs or on very few data points.

    There are all kinds of techniques developed to handle this kind of "multi fidelity Bayesian optimization" - in particular, they start evaluating $f(x^\prime)$ for our new point with low fidelity, and "abort" it if it's not looking good.

## Our idea - `SeqBayesOpt`

At a high-level, the idea here is to
  - Use a deep sequence model to model the posterior for $f$
  - Use the ability of a deep sequence model to "simulate out" later steps to pick the next step in a more principled fashion

The "sequence" in this idea would be the sequence of points $(x_1, f(x_1)), (x_2, f(x_2)), \cdots$ in our Bayesian optimization algorithm.

In practice, at any given step of the algorithm, we would have observed a sequence of points of values $(x_1, f(x_1)), \cdots, (x_n, f(x_n))$. The idea would be - for every $x^\prime_{n+1}$ - to autoregressively generate the full sequence of points and values we *would* obtain if we picked $x^\prime_{n+1}$ as our next point, and then use this to pick the best $x^\prime_{n+1}$. [Hong said something about finding the derivative of our final objective with respect to $x^\prime_{n+1}$ in "policy gradient" style and I'm fuzzy on the details, but will flesh this out down the road.]

It remains to determine how to generate sequences to *train* this sequence model on.
  - Each sequence will use one single function sampled from a Gaussian process...
      * In a v0, we would use the same GP to sample the function for every sequence
      * In a v1, we could first sample the mean and covariance function in some way, and then sample the function from a GP with that mean and covariance function
  - Once we have a function $f$, we need to decide what points $x_1, x_2, \ldots$ to select from the sequence. Two ways to do this
      * In a v0, pick them randomly
      * In a v1, pick $x_1$ randomly, and then pick the remaining $x$ by applying an existing "classical" Bayesian Optimization algorithm to $f$. (Or alternatively, applying Gradient Descent assuming full knowledge of $f$) [**IMPORTANT** if we do this, the sequence will no longer be exchangeable, because every $x$ will depend on all the ones before it - we sweep this under the rug for now.]

## The moonshot for the future - `Multifidelity SeqBayesOpt`

If the first idea above works, the next "moonshot" idea would be to use this technique for **multifidelity** Bayesian optimization. In particular, we would want our sequence to comprise the full multi-fidelity curve (in Hyperparameter Optimization, this is the full learning curve) for every value of $x$. It's unclear exactly what this would look like as a sequence model, but it's worth thinking about in the future.
