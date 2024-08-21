# Optimization algorithm ideas

## `Tadpole`

At a high-level, `tadpole` `involves using a sequence model as a "drop in" replacement for Gaussian Processes/Random forests/Parzen estimators/whatever in Bayesian Optimization.

At a high level, Bayesian optimization requires us, at every step, to
  - Get an estimate of the value of the function $f(x)$ to be optimized at a grid of points $x_1, x_2, \cdots, x_n$
  - Use those estimates to figure out which of those points to probe next

In its simplest form, the “sequence” in this idea would be the sequence of points and predictions - $(x_1, f(x_1)), (x_2, f(x_2)), \cdots (x_n, f(x_n))$ at every step. This sequence is exchangeable because the $x$ values are chosen on a grid.

Alternatively, for each point, the sequence could be repeated predictions of the function’s value at that point - $(x_1, f(x_1) + \epsilon_1), (x_1, f(x_1) + \epsilon_2), \cdots$ which could be averaged to get a single prediction at that point, where $\epsilon$ is exogenous noise.

We could generate a large number of such sequences by (1) picking a bunch of random mean and variance functions (2) generating a sequence of points from a GP with these parameters.

Down the road, this could be generalized to multi-fidelity optimization/training by making every point a full training sequence $(x_1, f_1(x_1), f_2(x_1), f_3(x_1)), \cdots, (x_2, f_1(x_2), f_2(x_2), f_3(x_2)), \cdots$

**What this would look like in practice**: at every step in our Bayes opt algo, we would consider a grid of points, get a prediction of $f(x)$ for every one of these points, and use expected improvement (or some other acquisition function) to pick the one to probe next.

## `Frog`

In this idea, the "sequence" is the sequence of points we probe in our Bayesian optimization process together with the value of the function we observe at each point.

This looks very similar to the above except that the points $x_1, x_2, \cdots, x_n$ would **not** be chosen on a grid. They would be the successive points in a Bayesian optimization process.

There are two big difficulties:
  - The sequence would no longer be exchangeable, because every point would be picked based on everything we’ve seen so far.
  - It’s unclear how we would generate the sequences to train our DNN on.

One idea would be to generate functions as in idea 1(a) and run “classical” Bayes Opt algorithms on them. The primary benefit of our method is that a forward pass through our DNN might be faster than doing posterior inference on a GP?
Another idea would be to generate functions as in idea 1(a) and run gradient descent on them. Maybe this “learnt” gradient descent would transfer over to the problems we’re working on.

**What this would look like in practice**: given past steps in a Bayesian opt search, we would just generate the rest of the path, and pick the next point in the path to evaluate. (Note: this wouldn’t make use of the full generated sequence, and wouldn’t make use of repeated observations of the sequence, so maybe not?)

## `Princess`

`Princess` moves to multi-fidelity optimization. In this idea, our sequence is the “full learning curve” - which again, is not exchangeable. Assuming we get past that hurdle, the idea would be to train a model that can start with something about the problem (eg: hyperparameters) and part of the learning curve, and then autoregressively generate the rest of the learning curve.

**What this would look like in practice**: for every possible next point in a Bayesian opt problem, generate the learning curve. Pick the combination with the best learning curve. Unclear to me how this would use the history of the Bayesian opt trajectory.