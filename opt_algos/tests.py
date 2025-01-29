# This files runs tests with all the optimizers on all the benchmarks

import os
import sys
import pickle

# Add the root directory to the path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from opt_algos import benchmarks
from opt_algos import optimizers

# Set a default seed for reproducibility
SEED = 38
# SEED = 41

benchmarks = {"DataModel": benchmarks.DataModelBenchmark}

optimizers = {
    "SMAC": optimizers.SMACOptimizer,
    "RandomSearch": optimizers.RandomSearchOptimizer,
    "GridSearch": optimizers.GridSearchOptimizer,
}

# These are times are chosen (by trial and error) to get the number of budget evaluations
# for SMAC roughly equal to 1000, which is what is used for the other techniques
smac_times = {"Branin": 1.5, "SimpleMixture": 10, "Lemur": 2, "DataModel": 60}
    
# Run

out = {}
        
for b in benchmarks:
    for o in optimizers:
        print(b, o)

        if o == "SMAC":
            max_time = smac_times[b]
        else:
            max_time = 16000

        benchmark = benchmarks[b]()

        out[(b, o)] = optimizers[o](
            benchmark.func, benchmark.search_space, benchmark.budget_space, max_time, seed=SEED
        ).minimize()

# Save the results
pickle.dump(out, open("tests.pickle", "wb"))
