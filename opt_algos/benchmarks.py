"""
This file contains benchmarks. For each benchmark, we create a class with a function
func, which can be fed into an optimizer (see optimizers.py for a set of requirements
this function should meet
"""

import data_model as dm
import numpy as np
import pandas as pd
import sklearn.ensemble as sk_e
import sklearn.metrics as sk_m
import os
import torch

class SimpleMixtureBenchmark:
    """
    This SimpleMixtureBenchmark involves a function z=f(x,y). The "true" function is a
    Branin function (https://www.sfu.ca/~ssurjano/branin.html), and we add normal noise
    with a variance of 120.

    The domain of x will be [-5, 10] and the domain of y will be [0, 15].

    We generate 8 "synthetic" datasets - datasets A-D will be generated as above, and
    datasets E-H will shuffle the z column and multiply it by 20, thus making these
    datasets completely useless for training.

    Each dataset will contain different ranges of X and Y

                    | X \in [-5, 2.5]  |  X \in [2.5, 10]  |
    ------------------|------------------|-------------------|
    Y \in [0, 7.5]    |     A and E      |      B and F      |
    Y \in [7.5, 15]   |     C and G      |      D and H      |
    ----------------------------------------------------------

    Thus, an optimal mix for training would be 0.25 each of A, B, C, and D.

    Dataset I will contain the full range of X and Y and not be shuffled - it will act as
    a validation set.
    """

    def __init__(self):
        # Generate the data
        self.generate_data()

        # Create a dictionary that can store "half-trained" models
        self.trained_models = {}

        # Set a seed
        self.seed = 123

        # Set the search space and budget space
        self.search_space = [[1, 100] for i in range(8)]
        self.budget_space = [1, 30]

    def func(self, z, x):
        """
        This function takes a fidelity and a set of dataset weights, and returns the
        model's performance on the validation set I for a successive fidelities.
        """

        # Normalize the weight variables
        x = [i / sum(x) for i in x]

        # Convert this configuration to a string so we can look it up in the trained
        # model dictionary
        x_s = str(x)

        # Check whether we already have a trained model - if so retrieve it, if not
        # create a new one
        if x_s in self.trained_models:
            m = self.trained_models[x_s]
        else:
            m = sk_e.RandomForestRegressor(
                n_estimators=0, random_state=self.seed, max_depth=6, warm_start=True
            )
            self.trained_models[x_s] = m

        # If the model has been trained before, we should only be seeing it again with a
        # higher budget (if the model was just created it'll have been created with a budget
        # of 0, so this will be true)
        assert z >= m.n_estimators

        # Get the data we're training on
        # ------------------------------
        this_data = []
        for ds, this_x in zip(["A", "B", "C", "D", "E", "F", "G", "H"], x):
            this_data.append(
                self.data[ds].sample(
                    int(len(self.data[ds]) * this_x), random_state=self.seed
                )
            )
        this_data = pd.concat(this_data)

        # Train and store intermediate results
        # ------------------------------------
        if m.n_estimators == z:
            print(f"WARNING: evaluating func at a repeat value of z (x = {x}, z = {z})")
            return [
                [
                    z,
                    sk_m.mean_squared_error(
                        self.data["I"]["Z"], m.predict(self.data["I"][["X", "Y"]])
                    ),
                ]
            ]

        out = []
        while m.n_estimators < z:
            m.n_estimators += 1
            m.fit(X=this_data[["X", "Y"]], y=this_data["Z"])
            out.append(
                [
                    m.n_estimators,
                    sk_m.mean_squared_error(
                        self.data["I"]["Z"], m.predict(self.data["I"][["X", "Y"]])
                    ),
                ]
            )

        # Return
        # ------
        return out

    @staticmethod
    def _branin(x, y):
        """
        Return the branin function value at x, y, which we use as our data generating
        pattern
        """

        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)

        return (y - b * x**2 + c * x - r) ** 2 + s * (1 - t) * np.cos(x) + s

    def generate_data(self):
        """
        Generate the 9 datasets as described above. This function returns a dict out
        with entries A through I constaining the datasets described above.
        """

        out = {}

        # Datasets A-H
        # ------------

        x = np.linspace(-5, 10, 200)
        y = np.linspace(0, 15, 200)
        X, Y = [i.ravel() for i in np.meshgrid(x, y)]
        Z = self._branin(X, Y) + np.random.normal(0, 120, len(X))
        df = pd.DataFrame(np.vstack([X, Y, Z]).T, columns=["X", "Y", "Z"])

        out["A"] = df[(df.X <= 2.5) & (df.Y <= 7.5)]
        out["B"] = df[(df.X > 2.5) & (df.Y <= 7.5)]
        out["C"] = df[(df.X <= 2.5) & (df.Y > 7.5)]
        out["D"] = df[(df.X > 2.5) & (df.Y > 7.5)]

        for i, j in zip(["A", "B", "C", "D"], ["E", "F", "G", "H"]):
            out[j] = out[i].copy()
            out[j].Z = np.random.choice(out[j].Z, len(out[j]), replace=False) * 20

        # Dataset I
        # ---------

        x = np.linspace(-5, 10, 100)
        y = np.linspace(0, 15, 100)
        X, Y = [i.ravel() for i in np.meshgrid(x, y)]
        Z = self._branin(X, Y) + np.random.normal(0, 120, len(X))
        out["I"] = pd.DataFrame(np.vstack([X, Y, Z]).T, columns=["X", "Y", "Z"])

        # Store the data
        # --------------
        self.data = out


class ConstantFunctionBenchmarkMixin:
    """
    This mix-in class is used to create benchmarks based on a constant function. The
    benchmark class should inherit from this class and implement a _raw_func(z, x)
    method that returns a *single* value - the function evaluated at z and x.

    This class will then handle the memo-ization of past value, and the returning of a
    full trace, as is required by the optimizers
    """

    def __init__(self):
        # Create a dictionary that can store past evaluations
        self.past_evaluations = {}

    def func(self, z, x):
        # Check whether we've already evaluated the function at this point
        x_s = str(x)

        if x_s not in self.past_evaluations:
            z_start = 0
        else:
            z_start = self.past_evaluations[x_s][0]

        if z_start == z:
            print(f"WARNING: evaluating func at a repeat value of z (x = {x}, z = {z})")
            return [self.past_evaluations[x_s]]

        out = [[zz, self._raw_func(zz, x)] for zz in range(z_start + 1, z + 1)]

        self.past_evaluations[x_s] = out[-1]

        return out


class BraninBenchmark(ConstantFunctionBenchmarkMixin):
    """
    Based on https://github.com/dragonfly/dragonfly/blob/master/examples/synthetic/branin/branin.py
    Modified to make fidelities integer values, and also to ensure that the fidelity is
    just one number

    In this example, func is simply a multi-fidelity version of a two-dimensional branin
    function https://www.sfu.ca/~ssurjano/branin.html
    """

    def __init__(self):
        # Set the search space and budget space
        self.search_space = [[-5, 10], [0, 15]]
        self.budget_space = [1, 100]

        # Call the parent __init__
        super().__init__()

    @staticmethod
    def _branin_with_params(x, a, b, c, r, s, t):
        """
        Computes the vanilla function
        """

        x1 = x[0]
        x2 = x[1]
        neg_ret = float(
            a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        )
        return -neg_ret

    def _raw_func(self, z, x):
        """
        Branin with Multi-fidelity.
        """

        # The initial implementation is designed for z to be between 0 and 1
        z /= 100

        b0 = 5.1 / (4 * np.pi**2)
        c0 = 5 / np.pi
        t0 = 1 / (8 * np.pi)
        delta_b = 0.01
        delta_c = 0.1
        delta_t = -0.005
        # Set parameters
        a = 1
        b = b0 - (1.0 - z) * delta_b
        c = c0 - (1.0 - z) * delta_c
        r = 6
        s = 10
        t = t0 - (1.0 - z) * delta_t
        return -BraninBenchmark._branin_with_params(x, a, b, c, r, s, t)


class LemurBenchmark(ConstantFunctionBenchmarkMixin):
    """
    This is a completely theoretical function we have designed for the purpose of testing
    BayesOpt algos.

    See readme file for details
    """

    def __init__(self):
        np.random.seed(123)

        # Number of datasets
        self.I = 8

        # Number of "underlying" attributes
        self.K = 5

        # Generate of distribution of attributes over datasets
        # ----------------------------------------------------
        self.x = []

        # Dataset 1 contains lots of attribute 1 and 2
        self.x.append([0.5, 0.5] + [0] * (self.K - 2))

        # Dataset 2 contains lots of attribute 3
        self.x.append([0] * 2 + [0.5] + [0] * (self.K - 3))

        # Dataset 3 contains the remaining attributes save for the last
        self.x.append([0] * 3 + [0.5] * (self.K - 4) + [0])

        # The remaining datasets contain only the last attribute
        for i in range(3, self.I):
            this_x = [0] * (self.K - 1)
            this_x.append(1)
            self.x.append(this_x)

        # Generate the baseline errors
        self.E = 1
        self.B = [np.random.uniform(0, 100) for k in range(self.K)]

        # Generate betas
        self.beta = [np.random.uniform(0.3, 0.8) for i in range(self.I)]

        # Set the search space and budget space
        self.search_space = [[1, 100] for i in range(self.I)]
        self.budget_space = [1, 50]

        super().__init__()

    def _raw_func(self, z, x):
        # Normalize the weight variables
        x = [i / sum(x) for i in x]

        # Get the loss
        out = self.E

        for k in range(self.K):
            this_out = self.B[k]

            this_divisor = 0
            for i in range(self.I):
                this_divisor += (x[i] * self.x[i][k] * z) ** self.beta[i]

            out += this_out / this_divisor

        return out

class DataModelBenchmark(ConstantFunctionBenchmarkMixin):
    """Create a data model benchmark that is compatible with 1D and 2D fidelity space"""

    def __init__(self, metric_index=4, device=None):
        """Metric index defines which metric to use for the data model"""
        self.metric_index = metric_index
        self.device = device
        
        # Load model and normalization stats using the correct path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(base_dir, "data_models", "20250119_174726_j8mad2i5")
        self.model, self.norm_stats = dm.load_model_for_prediction(checkpoint_path, device=self.device)

        # Set the search space and budget space
        # Logits for the 5 categories
        self.search_space = [[-3.0, 3.0] for i in range(5)]
        self.budget_space = [1, 196]

        super().__init__()

        feature_names = {
            0: "RedPajamaWikipedia",
            1: "RedPajamaStackExchange",
            2: "RedPajamaGithub",
            3: "RedPajamaArXiv",
            4: "RedPajamaBook",
            5: "Model Size (M)",
            6: "d_model",
            7: "Num Heads",
            8: "Training Steps",
        }
        METRIC_NAMES = {
            0: "Train Cross Entropy",
            1: "Common Crawl Cross Entropy",
            2: "C4 Cross Entropy",
            3: "Wikipedia Cross Entropy",
            4: "Stack Exchange Cross Entropy",
            5: "Github Cross Entropy",
            6: "ArXiv Cross Entropy",
            7: "Book Cross Entropy",
            8: "Hellaswag Accuracy",
            9: "PIQA Accuracy",
            10: "ARC Easy Accuracy",
        }

        print(
            f"Instantiating benchmark with y=metric {METRIC_NAMES[self.metric_index]}"
        )

    def _raw_func(self, z, x):
        """Evaluate data model at step z and data mixture x"""

        model_x = np.zeros(9)
        proportions = np.exp(x) / np.sum(np.exp(x))
        model_x[0:5] = proportions
        model_x[8] = 100 * z  # Training steps

        # Set other features to 1B features
        model_x[5] = 1000  # Model size in millions
        model_x[6] = 2048  # d_model dimension
        model_x[7] = 16  # Number of attention heads

        pred = dm.predict(
            self.model, model_x.reshape(1, -1), norm_stats=self.norm_stats
        )
        value = pred.squeeze()[self.metric_index]
        
        # Negate cross entropy metrics since we want to maximize performance
        if self.metric_index <= 7:  # Cross entropy metrics
            return value
        else:  # Accuracy metrics
            return -value  # Negate since we want to maximize performance


class NewDataModelBenchmark(ConstantFunctionBenchmarkMixin):
    """Create a data model benchmark that is compatible with 1D and 2D fidelity space"""

    def __init__(self, metric_index=4, device=None):
        """Metric index defines which metric to use for the data model"""
        self.metric_index = metric_index
        self.device = device
        
        # Load model and normalization stats using the correct path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(base_dir, "data_models", "20250119_174726_j8mad2i5")
        self.model, self.norm_stats = dm.load_model_for_prediction(checkpoint_path, device=self.device)

        # Set the search space and budget space
        # Logits for the 5 categories
        self.search_space = [[-3.0, 3.0] for i in range(5)]
        self.budget_space = [1, 196]

        super().__init__()

        feature_names = {
            0: "RedPajamaWikipedia",
            1: "RedPajamaStackExchange",
            2: "RedPajamaGithub",
            3: "RedPajamaArXiv",
            4: "RedPajamaBook",
            5: "Model Size (M)",
            6: "d_model",
            7: "Num Heads",
            8: "Training Steps",
        }
        METRIC_NAMES = {
            0: "Train Cross Entropy",
            1: "Common Crawl Cross Entropy",
            2: "C4 Cross Entropy",
            3: "Wikipedia Cross Entropy",
            4: "Stack Exchange Cross Entropy",
            5: "Github Cross Entropy",
            6: "ArXiv Cross Entropy",
            7: "Book Cross Entropy",
            8: "Hellaswag Accuracy",
            9: "PIQA Accuracy",
            10: "ARC Easy Accuracy",
        }

        print(
            f"Instantiating benchmark with y=metric {METRIC_NAMES[self.metric_index]}"
        )

    def _raw_func(self, size, steps, x):
        """Evaluate data model at step z and data mixture x"""

        model_x = np.zeros(9)
        proportions = np.exp(x) / np.sum(np.exp(x))
        model_x[0:5] = proportions
        model_x[8] = 100 * steps  # Training steps

        # Set other features to 1B features
        model_x[5] = size  # Model size in millions
        model_x[6] = 2048  # d_model dimension
        model_x[7] = 16  # Number of attention heads

        pred = dm.predict(
            self.model, model_x.reshape(1, -1), norm_stats=self.norm_stats
        )
        value = pred.squeeze()[self.metric_index]
        
        # Negate cross entropy metrics since we want to maximize performance
        if self.metric_index <= 7:  # Cross entropy metrics
            return value
        else:  # Accuracy metrics
            return -value  # Negate since we want to maximize performance