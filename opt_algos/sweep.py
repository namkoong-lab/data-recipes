# This files runs tests with all the optimizers on all the benchmarks

import os
import sys
import pickle
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import shutil
import datetime

# Add the root directory to the path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from opt_algos import benchmarks
from opt_algos import optimizers

# Create results directory structure
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", timestamp)
os.makedirs(results_dir, exist_ok=True)

# Create subdirectories for different types of outputs
logs_dir = os.path.join(results_dir, "logs")
plots_dir = os.path.join(results_dir, "plots")
data_dir = os.path.join(results_dir, "data")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Define the metrics to test
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
    10: "ARC Easy Accuracy"
}

# Configuration
SEEDS = [41, 38, 37 , 36, 35]  # Multiple seeds for robustness
METRICS = [1,2,3,4,5,6,7,8,9,10]  # Test Wikipedia CE, Stack Exchange CE, and Hellaswag Accuracy

# These are times are chosen (by trial and error) to get the number of budget evaluations
# for SMAC roughly equal to 1000, which is what is used for the other techniques
smac_times = {"DataModel": 60}  # Doubled from original
random_grid_time = 16000
optimizers = {
    "SMAC": optimizers.SMACOptimizer,
    "RandomSearch": optimizers.RandomSearchOptimizer,
    "GridSearch": optimizers.GridSearchOptimizer,
}

# Define colors for each algorithm
COLORS = {
    'SMAC': 'red',
    'RandomSearch': 'blue',
    'GridSearch': 'green'
}

def create_optimization_plot(results, metric):
    """Create a plot showing optimization progress for a given metric across all seeds"""
    plt.figure(figsize=(12, 8))
    
    # For each optimizer
    for opt_name in optimizers:
        # Collect all runs and convert to same index basis
        all_runs_values = []
        
        for seed in SEEDS:
            df = results[(opt_name, metric, seed)]['history']
            
            # Convert history to list of values if it's not already
            if isinstance(df['history'].iloc[0], str):
                df['history'] = df['history'].str.split('|').apply(lambda x: [tuple(map(float, i.split(':'))) for i in x])
            
            # Get the cumulative best value at each evaluation
            values = []
            if metric > 7:  # For accuracy metrics, track maximum
                current_best = float('-inf')
                for row in df.itertuples():
                    for _, val in row.history:
                        val = -val  # Convert back to positive accuracy immediately
                        current_best = max(current_best, val)
                        values.append(current_best)
            else:  # For cross entropy metrics, track minimum
                current_best = float('inf')
                for row in df.itertuples():
                    for _, val in row.history:
                        current_best = min(current_best, val)
                        values.append(current_best)
            
            all_runs_values.append(values)
        
        # Convert to numpy array and pad shorter runs to match longest
        max_len = max(len(run) for run in all_runs_values)
        padded_runs = []
        for run in all_runs_values:
            if len(run) < max_len:
                # Pad with the last value
                padded_runs.append(run + [run[-1]] * (max_len - len(run)))
            else:
                padded_runs.append(run)
        
        # Convert to numpy array for averaging
        all_runs_array = np.array(padded_runs)
        
        # Calculate mean and std across seeds
        mean_values = np.mean(all_runs_array, axis=0)
        std_values = np.std(all_runs_array, axis=0)
        
        # For accuracy metrics, we've already negated the values above, so no need to do it here
        # Remove the negation here since we already did it when collecting values
        # if metric > 7:
        #     mean_values = -mean_values
        #     std_values = std_values  # std doesn't need to be negated
        
        # Create x-axis values (evaluation indices)
        x = np.arange(len(mean_values))
        
        # Plot mean line
        plt.plot(x, mean_values, 
                color=COLORS[opt_name],
                label=opt_name,
                linewidth=2)
        
        # Add shaded region for standard deviation
        plt.fill_between(x, 
                        mean_values - std_values,
                        mean_values + std_values,
                        color=COLORS[opt_name],
                        alpha=0.2)
    
    plt.title(f'Algorithm Comparison - {METRIC_NAMES[metric]}\nSolid lines show mean best value across {len(SEEDS)} seeds')
    plt.xlabel('Number of Evaluations')
    
    if metric <= 7:  # Cross entropy metrics
        plt.ylabel('Cross Entropy (lower is better)')
        plt.yscale('log')
    else:  # Accuracy metrics
        plt.ylabel('Accuracy (higher is better)')
        plt.yscale('linear')  # Use linear scale for accuracy
    
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    return plt.gcf()

def run_experiment(metric_index, seed):
    """Run a single experiment with given metric and seed"""
    results = {}
    benchmark = benchmarks.DataModelBenchmark(metric_index=metric_index)
    
    for opt_name, opt_class in optimizers.items():
        print(f"Running {opt_name} with metric {METRIC_NAMES[metric_index]} (seed {seed})")
        
        max_time = smac_times["DataModel"] if opt_name == "SMAC" else random_grid_time
        
        # Create optimizer with log file in the logs directory
        log_file = os.path.join(logs_dir, f"{opt_name}_{METRIC_NAMES[metric_index].replace(' ', '_')}_{seed}.csv")
        optimizer = opt_class(
            benchmark.func, 
            benchmark.search_space, 
            benchmark.budget_space, 
            max_time, 
            seed=seed,
            log_file=log_file  # Pass the log file path to the optimizer
        )
        
        x, func_value, df, _ = optimizer.minimize()  # We don't need the log_file return value
        
        # For accuracy metrics (index > 7), we want the maximum value
        if metric_index > 7:
            min_idx = df['func'].idxmax()  # Use idxmax for accuracy metrics
            min_value = df.loc[min_idx, 'func']
            min_x = np.array([float(x) for x in df.loc[min_idx, 'x'].split('|')])
        else:
            min_idx = df['func'].idxmin()  # Use idxmin for cross entropy metrics
            min_value = df.loc[min_idx, 'func']
            min_x = np.array([float(x) for x in df.loc[min_idx, 'x'].split('|')])
        
        results[(opt_name, metric_index, seed)] = {
            'best_x': min_x,
            'best_value': min_value,
            'history': df,
            'log_file': log_file
        }
    
    return results

# Run experiments and collect results
all_results = {}
summary_stats = defaultdict(lambda: defaultdict(list))
best_configs = defaultdict(lambda: defaultdict(list))

for metric in METRICS:
    print(f"\nTesting metric: {METRIC_NAMES[metric]}")
    for seed in SEEDS:
        results = run_experiment(metric, seed)
        all_results.update(results)
        
        # Collect best values and configurations
        for opt_name in optimizers:
            best_value = results[(opt_name, metric, seed)]['best_value']
            best_x = results[(opt_name, metric, seed)]['best_x']
            summary_stats[metric][opt_name].append(best_value)
            best_configs[metric][opt_name].append(best_x)

# Calculate summary statistics and create plots
final_stats = {}
figures = {}

for metric in METRICS:
    final_stats[metric] = {}
    print(f"\nResults for {METRIC_NAMES[metric]}:")
    
    # Create visualization for this metric
    fig = create_optimization_plot(all_results, metric)
    figures[metric] = fig
    
    for opt_name in optimizers:
        best_values = summary_stats[metric][opt_name]
        configs = best_configs[metric][opt_name]
        
        mean_value = np.mean(best_values)
        std_value = np.std(best_values)
        
        # Find the overall best configuration
        best_idx = np.argmin(best_values)
        overall_best_value = best_values[best_idx]
        overall_best_config = configs[best_idx]
        
        final_stats[metric][opt_name] = {
            'mean_best': mean_value,
            'std_best': std_value,
            'best_values': best_values,
            'best_configs': configs,
            'overall_best_value': overall_best_value,
            'overall_best_config': overall_best_config
        }
        
        print(f"\n{opt_name}:")
        print(f"  Mean best value: {mean_value:.4f} ± {std_value:.4f}")
        print(f"  Overall best value: {overall_best_value:.4f}")
        print(f"  Best config: {overall_best_config}")

# Save all results with proper directory structure
output = {
    'all_results': all_results,
    'summary_stats': final_stats,
    'config': {
        'seeds': SEEDS,
        'metrics': METRICS,
        'metric_names': METRIC_NAMES
    }
}

# Save main results file
results_file = os.path.join(data_dir, "optimization_results.pickle")
with open(results_file, 'wb') as f:
    pickle.dump(output, f)

# Save plots
for metric, fig in figures.items():
    # Save as PNG
    plot_file = os.path.join(plots_dir, f"optimization_plot_{METRIC_NAMES[metric].replace(' ', '_')}.png")
    fig.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory
    print(f"\nPlot for {METRIC_NAMES[metric]} saved to {plot_file}")

# Create a README file in the results directory
readme_content = f"""Optimization Results - {timestamp}

Configuration:
- Seeds: {SEEDS}
- Metrics: {[METRIC_NAMES[m] for m in METRICS]}
- Optimizers: {list(optimizers.keys())}

Directory Structure:
- data/: Contains the main results pickle file
- logs/: Contains individual optimization logs
- plots/: Contains interactive visualization plots

Results Summary:
"""

# Add summary statistics to README
for metric in METRICS:
    readme_content += f"\n{METRIC_NAMES[metric]}:\n"
    for opt_name in optimizers:
        stats = final_stats[metric][opt_name]
        readme_content += f"- {opt_name}:\n"
        readme_content += f"  Mean best value: {stats['mean_best']:.4f} ± {stats['std_best']:.4f}\n"
        readme_content += f"  Overall best value: {stats['overall_best_value']:.4f}\n"
        readme_content += f"  Best config: {stats['overall_best_config'].tolist()}\n"

with open(os.path.join(results_dir, "README.md"), 'w') as f:
    f.write(readme_content)

print(f"\nAll results saved in: {results_dir}")
print("Directory structure:")
print(f"- Main results: {results_file}")
print(f"- Logs: {logs_dir}")
print(f"- Plots: {plots_dir}")
print("See README.md in results directory for detailed information.")
