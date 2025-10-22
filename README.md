# Data Mixture Optimization: A Multi-fidelity Multi-scale Bayesian Framework

Careful curation of data sources can significantly improve the performance of LLM pre-training, but
predominant approaches rely heavily on intuition or costly trial-and-error, making them difficult to generalize across different data domains and downstream tasks.
Although scaling laws can provide a principled and general approach for data curation, standard deterministic extrapolation from small-scale experiments to larger scales requires strong assumptions on the reliability of such extrapolation, whose brittleness has been highlighted in prior works.
In this paper, we introduce a \emph{probabilistic extrapolation framework} for data mixture optimization that avoids rigid assumptions and explicitly models the uncertainty in performance across decision variables.
We formulate data curation as a sequential decision-making problem---multi-fidelity, multi-scale Bayesian optimization---where {data mixtures, model scale, training steps} are adaptively selected to balance training cost and potential information gain.
Our framework naturally gives rise to algorithm prototypes that leverage noisy information from inexpensive experiments to systematically inform costly training decisions.
To accelerate methodological progress, we build a simulator based on 472 language model pre-training runs with varying data compositions from the SlimPajama dataset.
We observe that even simple kernels and acquisition functions can enable principled decisions across training models from 20M to 1B parameters and achieve **2.6x** and **3.3x** speedups compared to multi-fidelity Bayesian optimization and random search baselines. Taken together, our framework underscores potential efficiency gains achievable by developing principled and transferable data mixture optimization methods.

## Access Training Run Results

The training results are in `results/data_mixing_runs.pkl`

One can access the results with the following code snippet:
```
import pickle

with open("data_mixing_runs.pkl", "rb") as f:
    runs = pickle.load(f)
```

## Access the Simulator

The simulator accepts the tuple {data_mixture, model_scale, training_steps} and return a prediction for the performance of a model trained with the corresponding configuration.
We offer the simulator to the community as a cheap surrogate testbed for data mixture optimization methodologies, avoiding the cost of actually training a language model

Note that we have only trained the simulator for 20M, 60M, 150M, 300M, 500M, 700M, and 1B scales.
From preliminary experiments, we found the simulator is unreliable out of distribution w.r.t model scales, and hence it predictions should only be trusted within these scales.

One can access the simulator `DataModelBenchmark` under `/opt_algos/benchmarks.py`. The `_raw_func_with_model_scale` method carries out the aforementioned predictions.
