'''
This file contains optimizers. For each optimizer, we create a class that should
be initialized with the following arguments
  - func : a function that accepts two arguments
             * z : an integer or float containing the budget/fideily we want to
                   evaluate the function with
             * x : an array containing the function parameter values
           
           This is the function to be optimized, at some value of z (say z_max).
           When evaluated at a value of z < z_max, the function returns some
           "less accurate" value of the function at that point. For example, z
           could correspond to the number of gradient descent steps taken during
           training.

           The function should perform its own "memoization" in some way - in
           other words, if the algorithm calls it with a given value of x and
           a fidelity of 10, and then calls it again with the SAME value of x
           and a fidelity of 20, it should internally be able to load a checkpoint
           from the previous call and use that to generate the higher fidelity
           reading.

           The function should return a list of tuples - each tuple will have the
           form [z', func(z', x)], where z' is some budget/fidelity <= z.
           
           If the function is being evaluated for the first time for a given x,
           the list will contain the function evaluated at every value of z from
           1 to z:
               [ [1, f(1, x)], [2, f(2, x)], [3, f(3, x)], ..., [z, f(z, x)] ]
            
           The next time the function is evaluated at a value z' > z, the list
           should contain the function evaluated at every value of z from z to z':
                [ [z, f(z, x)], [z+1, f(z+1, x)], ..., [z', f(z', x)] ]
           
           If the function is evaluated at a value of z that it has ALREADY been
           evaluated at before (which can happen because some algorithms will use
           decimal fidelities which need to be rounded - so for eg, fidelities of
           3.7 and 3.9 will look different to the algo, but will look the same for
           out function; see _func_wrapped in the Optimizer class), the function
           should return a single tuple with that value of z, and print a warning
           if desired.
                
           The algorithm will only use the LAST value in this list (i.e., the best
           fidelity), but we need all the other values for later plotting. If you
           do not intend to do any such plotting/visualization/analysis, the
           function can just return a list with a single tuple containing func
           evaluated at the largest value of z.
  - search_space : a list of lists with as many entries as dimensions in x (the
                   variable to be optimized over). Each list should contain two
                   values - the minimum and maximum value of that variable.
  - budget_space : a single list containing the minimum and maximum value of the
                   budget/fidelity parameter z.
  - max_time : the maximum amount of time the optimizer is allowed to run;
               unfortunately, the implementation of this is likely to be quite
               algorithm-dependent in terms of how they count "time". See the
               docstring at the start of each class.
  - seed : the seed for random number generation
  - log_file : the path to the log file for storing function evaluations

The minimize() method then runs the relevant optimization algorithm and returns a
tuple with the following:
  - The optimal x value returned by the optimizer
  - The value of the function at that value of x and its highest fidelity
  - A Pandas dataframe with one row per call to the function that is being
    optimized, in the order in which they were called by the algo, with the
    following columns
      * z : the value of z with which the function was called
      * func : the value of func(z, x)
      * history : the full history returned by func (see documentation for func above)
        as a list
      * x : the value of x at which the funtion was optimized, as a list
  - The name of the log file in which the dataframe was stored, for future
    reference

In addition to the log file, it's possible that the optimizers themselves will
spew out various files/folders
'''

import os
import numpy as np
import pandas as pd

class Optimizer():
    def __init__(self, func, search_space, budget_space, max_time, seed=42, log_file=None):
        self.func = func
        self.search_space = search_space
        self.budget_space = budget_space
        self.max_time = max_time
        self.seed = seed  # Store the seed
        self.log_file = log_file  # Store the log file path

        # Ensure the minimum budget is > 0. This is important because we will assume
        # the "cost" of evaluating func is equal to the budget; if we allow a budget
        # of 0, this would allow an infinite number of evaluations without increasing
        # the budget
        assert budget_space[0] > 0, 'Minimum budget must be > 0'

    def _func_wrapped(self, z, x):
        '''
        This function wraps the function to be optimized, but logs every evaluation
        to self.log_file. It returns the value of the HIGHEST fidelity function call
        '''

        # The algorithms we look at might sometime ask us to evaluate the function at
        # non-integer fidelities. In our application we only want to use at integer
        # fidelities (e.g., training steps) so we need to round up (to make sure that
        # fidelities < 1 get rounded up to 1)
        z = int(z) if z == int(z) else int(z) + 1
        
        # Get the full trace
        res = self.func(z, x)

        # Get the highest fidelity function value
        func = res[-1][1]

        # Get the full trace in the format z1:func(z1,x)|z2:func(z2,x)|... to be able
        # to log it to a CSV file
        history = '|'.join([f'{zz}:{ff}' for zz, ff in res])

        # Get x in the format x1|x2|...
        x_str = '|'.join([str(i) for i in x])

        with open(self.log_file, 'a') as f:
            f.write(f'{z},{func},{history},{x_str}' + '\n')

        return func

    def minimize(self):
        # If no log file specified, create one with timestamp
        if self.log_file is None:
            # Create a log file
            while (not self.log_file) or os.path.exists(self.log_file):
                # Get a log file name in the format log_classname_YYYY-MM-DD_HH-MM-SS.csv
                import datetime
                self.log_file = f'log_{self.__class__.__name__}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv'

        # Write the file headers
        with open(self.log_file, 'w') as f:
            f.write('z,func,history,x\n')

        # Minimize and get the optimum
        x = self._minimize()

        # Get the value of the function at the optimum
        func = self.func(self.budget_space[1], x)[-1][1]

        # Read the log
        df = pd.read_csv(self.log_file)

        # Convert the history column stored in the format x1:func(z1,x)|z2:func(z2,x)|...
        # into a list of tuples
        if isinstance(df['history'].iloc[0], str):
            df.history = df.history.str.split('|').apply(lambda x: [tuple(map(float, i.split(':'))) for i in x])
        
        # Return
        return x, func, df, self.log_file

class BOCAOptimizer(Optimizer):
    '''
    For the BOCA optimizer, the "time" in max_time is the cumulative sum of budgets
    that our function is evaluated on
    '''

    def _minimize(self):
        import dragonfly

        # BOCA passes an array for x and an array for z. Our function will expect
        # a scalar for z - wrap the function accordingly
        def this_func(z, x):
            return self._func_wrapped(z[0], x)

        # Optimize (the third argument output by minimise is the history, but we're
        # capturing that already in the log file)
        #
        # capital_type='return_value'  is what ensures the "time taken" is the cumulative
        # sum of budgets. Thre is also a "walltime" option, but for some reason it
        # requires the objective function to the pickle-able, which leads to all kinds of
        # pain because it means the function needs to be global to the whole module
        #
        # Note also that for whatever reason, the fidel_cost_func doesn't get passed the
        # actual fidelity value that gets calclated - instead, it gets passed a number
        # between 0 and 1 that represents the fraction of the budget search space that is
        # being used in that iteration (for eg: if budget is between 1 and 100, and we
        # want to evaluate the function at a budget of 20, the fidel_cost_func gets passed
        # a value of 0.2 - so we have to scale it)
        
        # TODO : there's a way to specify the fidelity space has to be integer; the trick
        # is somewhere in dragonfly.exd.domains.IntegralDomain, but it's unclear exactly
        # how to use it. See also this link which seems to show a way to do it
        # https://github.com/dragonfly/dragonfly/blob/master/examples/synthetic/hartmann6_4/in_code_demo.py
        min_val, min_pt, _ = dragonfly.minimise_multifidelity_function(
                                func            = this_func,
                                fidel_space     = [self.budget_space],
                                domain          = self.search_space,
                                fidel_to_opt    = self.budget_space[1],
                                fidel_cost_func = lambda z : self.budget_space[0] + self.budget_space[1] * z[0],
                                max_capital     = self.max_time,
                                capital_type    = 'return_value',
                                opt_method      = 'bo',
                                config          = None,
                                options         = None,
                                reporter        = 'default')
        
        return min_pt

class SMACOptimizer(Optimizer):
    '''
    For the SMAC optimizer, the "time" in max_time is the wall time
    '''
    def __init__(self, func, search_space, budget_space, max_time, seed=42, log_file=None, 
                 initial_design_size=10, eta=3):
        """
        Parameters:
            initial_design_size: Number of initial configurations to evaluate (default: 10)
            eta: Reduction factor for successive halving (default: 3)
                Controls bracket sizes - higher values = smaller brackets
        """
        super().__init__(func, search_space, budget_space, max_time, seed, log_file)
        self.initial_design_size = initial_design_size
        self.eta = eta

    def _minimize(self):
        import smac
        import ConfigSpace

        # Create the configuration space
        cs = ConfigSpace.ConfigurationSpace()
        for i, (min_val, max_val) in enumerate(self.search_space):
            cs.add_hyperparameter(ConfigSpace.Float(f'x{i}', [min_val, max_val]))
        
        # Create the scenario
        scenario = smac.Scenario(cs,
                                 walltime_limit = self.max_time,
                                 n_trials = float('inf'),
                                 min_budget = self.budget_space[0],
                                 max_budget = self.budget_space[1],
                                 n_workers = 1,
                                 seed = self.seed)

        # Use a hyperband intensifier with custom eta parameter
        intensifier = smac.intensifier.hyperband.Hyperband(
                                scenario,
                                incumbent_selection='highest_budget',
                                eta=self.eta,  # Control bracket reduction factor
                                seed=self.seed)
        
        # SMAC provides the optimization function three arguments - config, seed, and budget.
        # config is provided as a dictionary-like object, so we need to convert it
        def this_func(config, seed, budget, n_vars=len(self.search_space)):
            return self._func_wrapped(budget, np.array([config[f'x{i}'] for i in range(n_vars)]))

        # Create initial design with custom size
        initial_design = smac.initial_design.RandomInitialDesign(scenario, n_configs=self.initial_design_size)

        # Optimize
        smac_instance = smac.MultiFidelityFacade(
                                scenario,
                                this_func,
                                initial_design=initial_design,
                                intensifier=intensifier,
                                overwrite=True,
                                logging_level=None)
        res = smac_instance.optimize()

        # Extract the results
        return np.array([res[f'x{i}'] for i in range(len(self.search_space))])

class RandomSearchOptimizer(Optimizer):
    '''
    For the Random optimizer, the "time" in max_time is the number of function evaluations
    '''

    def _minimize(self):
        # Set the seed
        np.random.seed(self.seed)

        n_evals = int(np.round(self.max_time / self.budget_space[1]))

        min_func = np.inf
        min_x = None

        for i in range(n_evals):
            # Randomly sample from the search space
            x = np.array([np.random.uniform(min_val, max_val) for min_val, max_val in self.search_space])

            # Evaluate the function
            func = self._func_wrapped(self.budget_space[1], x)

            if func < min_func:
                min_func = func
                min_x = x
        
        return min_x

class GridSearchOptimizer(Optimizer):
    '''
    For the Grid optimizer, the "time" in max_time is the number of function evaluations
    (roughly - as much as possible given the number of dimensions)
    '''

    def _minimize(self):
        # Set the seed for reproducible random sampling if needed
        np.random.seed(self.seed)
        
        n_evals = int(np.round(self.max_time / self.budget_space[1]))

        # Figure out the number of points per dimension
        n_per_dim = int(np.ceil(n_evals ** (1/len(self.search_space))))

        # Create uniform grids for each dimension
        grids = [np.linspace(min_val, max_val, n_per_dim) for min_val, max_val in self.search_space]

        # Create a meshgrid to find all combinations of values
        mesh = np.meshgrid(*grids)
        points = np.stack(mesh, axis=-1).reshape(-1, len(self.search_space))

        # If we ended up with too many evaluation points, randomly choose some
        if len(points) > n_evals:
            points_to_pick = np.random.choice(len(points), n_evals, replace=False)
            points = [points[i] for i in range(len(points)) if i in points_to_pick]

        # Go through all grid points
        min_func = np.inf
        min_x = None

        for x in points:
            func = self._func_wrapped(self.budget_space[1], x)

            if func < min_func:
                min_func = func
                min_x = x
        
        return min_x