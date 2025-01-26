# Bayesian optimization testbed

This folder provides a Bayesian Optimization test bed. It defines two kinds of objects - benchmarks, which provide functions to be optimized, and optimizers, which actually do the optimization.

See `opt_ideas.md` for some other optimization algo ideas.

## Folder structure

This folder includes two files

- `optimizers.py` : contains one class per optimizer
- `benchmarks.py` : contains one class per benchmark

See the files for documentation.

In addition, the following two files can be used to demo everything

- `test.py` : runs every optimizer against every benchmark, and outputs the results into `tests.pickle`
- `visualize.ipynb` : loads `tests.pickle` and lets us visualize the optimization runs

The latter provides an interactively zoomable plot to explore the trace of an algorithm run.

### Optimizers

Right now, the following optimizers are implemented

- SMAC
- BOCA
- Random search
- Grid search

### Benchmarks

Right now, the following benchmarks are implemented

- A simple mixture benchmark
- The branin function
- Lemur (see below)

## Installing

The following commands, run in a row, made everything work on a Mac with an M3 chip:

```
pip install smac==2.2.0
pip install pandas==2.0.3
pip install matplotlib==3.7.5
pip install seaborn==0.13.2
pip install PyQt5==5.15.11
pip install plotly
pip install torch==2.2.2
pip install botorch
```


Notes

- To install `smac`, you need `swig` installed on your machine
- (not needed if dragonfly is not used)`numpy` needs to be instaleld twice because after installing smac, `numpy` will be upgraded to 1.24.4 - force it back to 1.21 with the command above, or else `dragonfly` will break

This isn't really a sustainable way to maintain dependencies - as we add more pacakges that we want to test with this repo, we are bound to encounter packages with conflicting dependencies. I suspect the solution will be to implement some of these algos ourselves rather than rely on packages.

## ToDo

- The BOCA optimizer is incredibly slow; figure out why
- Re-implement both optimizers ourselves rather than relying on packages which is quite opaque and sometimes rely on very old dependencies.
- Dig into the SMAC and BOCA code to figure out what they're actually doing, and figure out how to implement it ourselves directly. I already did a bunch of work on this for SMAC which I'm happy to discuss, but not BOCA. It's be great to rationalize some of what we see in the optimization traces.
- Investigate the best way to optimize over weights that sum to 1 - right now, we search over a the range 1-100 for every weight, and then divide each weight by the sum of the weights
- SMAC currently carries out many rounds of hyperband (each with new initial points). Investigate whether it might be better to carry out only one round starting with many more points.
- SMAC seems to wait for quite a long time before fitting a random forest - before then, all the points are selected randomly; consider playing around with this "warm up" period
- There seems to be no way to set "seeds" anywhere, so results are going to be different every time. Play with this.

## `LemurBenchmark` details

This section gives details of the `LemurBenchmark` in `benchmarks.py` - it's easier to do it here than in code because of equations.

The proposed benchmark (arbitrarily named `Lemur`) isn't based on any strong theoretical foundation. See the discussion [here](https://docs.google.com/document/d/1oWs8NmZvv4ONsZK42HyqG4vexymurk4BhpEB8kYxrso/edit). But it provides what we think are some desirably properties.

The idea is to hypothesize that there are $K$ “canonical ingredients” that make up the universe of datasets, and each dataset has these canonical ingredients in varying degrees. In fact, each dataset is characterized by a vector $x^{(i)} \in \mathbb{R}^K$ with $\sum x_k = 1$, which tells us how much of each "ingredient" each dataset contributes. (This is a poor man’s version of the "every dataset has an embedding" idea).

We assume that there are various kinds of "loss" that each "ingredient" addresses ($B_1, \cdots, B_K$), and that the more tokens with that “ingredient” are included, the more the loss is reduced.

Finally, we assume each dataset has a $\beta_i$ which gives us its coefficient in the power law - in other words, it tells us the speed at which the value of tokens from this dataset decline.

Let $D$ be the total number of tokens the model is trained on, $I$ be the total number of different datasets, and $w_i$ be the proportion of each dataset type, with $\sum w_i = 1$. Then we stipulate that the loss from a model trained with $D$ tokens with weights $w$ is

$$ L(D, w) = E + \sum*{k=1}^K \frac{B_k}{\sum*{i=1}^I (w_i x^{(i)}\_k D)^{\beta_i}} $$

In the above, $E$, $B_1, \cdots, B_K$, $x^{(1)}, \cdots, x^{(I)}$, and $\beta_1, \cdots, \beta_I$ are all parameters.
