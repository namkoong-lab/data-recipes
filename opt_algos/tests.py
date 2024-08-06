# This files runs tests with all the optimizers on all the benchmarks

import benchmarks
import optimizers
import pickle

benchmarks = {'Branin'         : benchmarks.BraninBenchmark       ,
              'SimpleMixture'  : benchmarks.SimpleMixtureBenchmark,
              'Lemur'          : benchmarks.LemurBenchmark         }

optimizers = {'SMAC'         : optimizers.SMACOptimizer         ,
              'BOCA'         : optimizers.BOCAOptimizer         ,
              'RandomSearch' : optimizers.RandomSearchOptimizer ,
              'GridSearch'   : optimizers.GridSearchOptimizer     }

# These are times are chosen (by trial an error) to get the number of budget evaluations
# for SMAC roughly equal to 1000, which is what is used for the other techniques
smac_times = {'Branin'        : 1.5,
              'SimpleMixture' : 10,
              'Lemur'         : 2   }

# Run

out = {}

for b in benchmarks:
    for o in optimizers:
        print(b, o)

        if o == 'SMAC':
            max_time = smac_times[b]
        else:
            max_time = 1000
        
        benchmark = benchmarks[b]()

        out[(b,o)] = optimizers[o](benchmark.func, benchmark.search_space, benchmark.budget_space, max_time).minimize()

# Save the results
pickle.dump(out, open('tests.pickle', 'wb'))