import os
import sys

# Add the root directory to the path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

import json
from datetime import datetime
from itertools import product
import torch
import numpy as np
from opt_algos.botorch.train_eikg import BayesianOptimizer, OptimizationConfig, AcquisitionType, METRIC_NAMES, DEVICE, tkwargs
from botorch.models.cost import AffineFidelityCostModel
from botorch import fit_gpytorch_mll
import matplotlib.pyplot as plt
import wandb

# Test run flag - set to True for quick testing
TEST_RUN = False

def convert_to_serializable(obj):
    """Convert numpy arrays and torch tensors to lists for JSON serialization."""
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def create_sweep_configs():
    """Define the parameter sweep configurations."""
    if TEST_RUN:
        sweep_params = {
            "initial_points": [4],
            "batch_size": [4],
            "fixed_cost": [5000.0],
            "n_iterations": [1],  # Just one iteration
            "metric_index": [4],  # Just Stack Exchange CE
            "fidelities": [
                torch.tensor([100.0, 150.0, 196.0], **tkwargs),  # Ensure fidelities are on correct device
            ]
        }
        print("\nRunning in TEST mode with minimal configuration")
    else:
        sweep_params = {
            "initial_points": [4],
            "batch_size": [4],# [4, 8],
            "fixed_cost": [10.0],# 100.0],
            "n_iterations": [8],
            "metric_index": [4, 7, 8],  # Stack Exchange CE, Hellaswag, PIQA, ARC Easy
            "fidelities": [
                # torch.tensor([100.0, 150.0, 196.0], **tkwargs),
                torch.tensor([50.0, 100.0, 150.0, 196.0], **tkwargs),
                # torch.tensor([25.0, 50.0, 100.0, 150.0, 196.0], **tkwargs)
            ]
        }
    
    print("\nSweeping over the following metrics:")
    for idx in sweep_params["metric_index"]:  # No need to access [0]
        print(f"  - {METRIC_NAMES[idx]}")
    
    # Generate all combinations
    param_names = sweep_params.keys()
    param_values = sweep_params.values()
    
    configs = []
    for values in product(*param_values):
        config_dict = dict(zip(param_names, values))
        
        # Create cost model
        cost_model = AffineFidelityCostModel(
            fidelity_weights={5: 1.0},
            fixed_cost=config_dict["fixed_cost"]
        )
        
        # Create optimization config
        opt_config = OptimizationConfig(
            metric_index=config_dict["metric_index"],  # No need to access [0]
            fidelities=config_dict["fidelities"],
            cost_model=cost_model
        )
        
        configs.append({
            "opt_config": opt_config,
            "params": config_dict
        })
    
    return configs

def run_sweep():
    """Run the parameter sweep experiments."""
    # Create base results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_results_dir = os.path.join('results', f'sweep_{timestamp}')
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Get all configurations
    configs = create_sweep_configs()
    print(f"\nRunning sweep with {len(configs)} configurations")
    
    # Store sweep metadata
    sweep_metadata = {
        "timestamp": timestamp,
        "total_configs": len(configs),
        "completed_configs": 0,
        "failed_configs": [],
        "metrics": {idx: METRIC_NAMES[idx] for idx in [configs[0]["params"]["metric_index"]]},
        "test_run": TEST_RUN,
        "configurations": convert_to_serializable([config["params"] for config in configs])
    }
    
    # Run each configuration
    for config_idx, config in enumerate(configs, 1):
        # Initialize wandb run for this configuration
        run_name = f"sweep_{timestamp}_config_{config_idx}"
        wandb.init(
            project="botorch",
            name=run_name,
            config={
                "sweep_timestamp": timestamp,
                "config_idx": config_idx,
                "metric": METRIC_NAMES[config['params']['metric_index']],
                "batch_size": config['params']['batch_size'],
                "fixed_cost": config['params']['fixed_cost'],
                "n_iterations": config['params']['n_iterations'],
                "initial_points": config['params']['initial_points'],
                "fidelities": config['params']['fidelities'].tolist(),
            },
            reinit=True  # Allow multiple runs in the same process
        )

        print(f"\n{'='*80}")
        print(f"Running configuration {config_idx}/{len(configs)}")
        print(f"Metric: {METRIC_NAMES[config['params']['metric_index']]}")
        print(f"Parameters: {config['params']}")
        
        # Create experiment directory with metric name
        metric_name = METRIC_NAMES[config['params']['metric_index']].lower().replace(' ', '_')
        exp_dir = os.path.join(base_results_dir, f'exp_{config_idx:03d}_{metric_name}')
        os.makedirs(exp_dir, exist_ok=True)
        
        try:
            # Initialize optimizer
            optimizer = BayesianOptimizer(config["opt_config"])
            
            # Generate initial data
            initial_train_x, initial_train_obj = optimizer.generate_initial_data(n=config["params"]["initial_points"])
            initial_best_value = initial_train_obj.max().item()
            
            # Store results for both acquisition functions
            results = {}
            
            for acq_type in [AcquisitionType.EI, AcquisitionType.KG]:
                print(f"\nRunning {acq_type.value} acquisition function")
                
                # Reset optimizer for each acquisition type
                optimizer = BayesianOptimizer(config["opt_config"])
                optimizer.history["acq_type"] = acq_type.value
                
                # Use the same initial data
                train_x = initial_train_x.clone()
                train_obj = initial_train_obj.clone()
                
                # Initialize history with initial point
                optimizer.history["costs"] = []  # Will add initial cost
                optimizer.history["best_values"] = []  # Will add initial best
                optimizer.history["recommended_values"] = []  # Will add initial recommendation
                
                # Get initial recommendation
                mll, model = optimizer.initialize_model(train_x, train_obj)
                fit_gpytorch_mll(mll)
                initial_rec = optimizer.get_recommendation(model)
                initial_raw_value = optimizer.problem.func(int(initial_rec[0, -1].item()), initial_rec[0, :-1].tolist())[-1][1]
                
                # Add initial points to history
                optimizer.history["costs"].append(0.0)
                optimizer.history["best_values"].append(initial_best_value)
                optimizer.history["recommended_values"].append(-initial_raw_value)  # Store initial recommendation
                
                # Add initial point to iteration results
                iteration_results = []
                iteration_results.append({
                    "iteration": 0,  # Mark as initial point
                    "best_value": initial_best_value,
                    "cumulative_cost": 0.0,
                    "training_points": len(train_x),
                    "recommended_value": -initial_raw_value
                })
                
                cumulative_cost = 0.0
                
                # Run optimization
                for iteration in range(config["params"]["n_iterations"]):
                    mll, model = optimizer.initialize_model(train_x, train_obj)
                    fit_gpytorch_mll(mll)
                    
                    acq_func = optimizer.get_acquisition_function(model, acq_type)
                    new_x, new_obj, cost = optimizer.optimize_and_get_observation(acq_func, acq_type)
                    
                    if torch.isnan(new_obj).any():
                        print("Warning: NaN values detected in observations")
                        continue
                    
                    train_x = torch.cat([train_x, new_x])
                    train_obj = torch.cat([train_obj, new_obj])
                    cumulative_cost += cost
                    
                    # Get recommendation for this iteration
                    mll, model = optimizer.initialize_model(train_x, train_obj)
                    fit_gpytorch_mll(mll)
                    rec = optimizer.get_recommendation(model)
                    raw_value = optimizer.problem.func(int(rec[0, -1].item()), rec[0, :-1].tolist())[-1][1]
                    
                    # Always append new values - don't try to update existing ones
                    current_best = train_obj.max().item()
                    optimizer.history["costs"].append(cumulative_cost.item())
                    optimizer.history["best_values"].append(current_best)
                    optimizer.history["recommended_values"].append(-raw_value)
                    
                    iteration_results.append({
                        "iteration": iteration + 1,
                        "best_value": current_best,
                        "cumulative_cost": cumulative_cost.item(),
                        "training_points": len(train_x),
                        "recommended_value": -raw_value
                    })
                
                # Get final recommendation and append it if the last cost was repeated
                final_rec = optimizer.get_recommendation(model)
                final_raw_value = optimizer.problem.func(int(final_rec[0, -1].item()), final_rec[0, :-1].tolist())[-1][1]
                if len(optimizer.history["costs"]) > len(optimizer.history["recommended_values"]):
                    optimizer.history["recommended_values"].append(-final_raw_value)
                
                # Store results
                results[acq_type.value] = {
                    "costs": convert_to_serializable(optimizer.history["costs"]),
                    "best_values": convert_to_serializable(optimizer.history["best_values"]),
                    "recommended_values": convert_to_serializable(optimizer.history["recommended_values"]),
                    "iterations": convert_to_serializable(iteration_results),
                    "final_recommendation": convert_to_serializable(final_rec),
                    "total_cost": cumulative_cost.item(),
                    "initial_recommendation": convert_to_serializable(initial_rec)  # Store initial recommendation
                }
                
                # Debug print before plotting
                print(f"\nBefore plotting {acq_type.value}:")
                print(f"History costs: {optimizer.history['costs']}")
                print(f"History best values: {optimizer.history['best_values']}")
                print(f"History recommended values: {optimizer.history['recommended_values']}")
            
            # Save experiment results
            exp_results = {
                "config": {
                    "params": convert_to_serializable(config["params"]),
                    "metric_name": METRIC_NAMES[config['params']['metric_index']],
                    "metric_index": config['params']['metric_index']
                },
                "initial_data": {
                    "best_value": initial_best_value,
                    "n_points": len(initial_train_x)
                },
                "results": results
            }
            
            with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
                json.dump(exp_results, f, indent=2)
            
            # Create comparison plot
            plt.clf()  # Clear any existing figures
            plt.close('all')  # Close all figures to free memory
            
            try:
                fig = plt.figure(figsize=(10, 6))
                for acq_type, data in results.items():
                    print(f"\nPlotting {acq_type} data:")
                    costs = data["costs"]
                    best_values = data["best_values"]
                    recommended_values = data["recommended_values"]
                    
                    # Ensure recommended_values has same length as costs by extending the last value
                    if len(recommended_values) < len(costs):
                        recommended_values = recommended_values + [recommended_values[-1]] * (len(costs) - len(recommended_values))
                    
                    print(f"After adjustment:")
                    print(f"Costs length: {len(costs)}")
                    print(f"Best values length: {len(best_values)}")
                    print(f"Recommended values length: {len(recommended_values)}")
                    
                    plt.plot(costs, best_values, 
                            marker='o', linestyle='--', label=f"{acq_type.upper()} (Observed)")
                    plt.plot(costs, recommended_values,
                            marker='s', label=f"{acq_type.upper()} (Recommended)")
                
                plt.xlabel("Cumulative Cost")
                plt.ylabel(f"Best {METRIC_NAMES[config['params']['metric_index']]}")
                plt.title(f"Comparison of Acquisition Functions\nMetric: {METRIC_NAMES[config['params']['metric_index']]}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                
                # Save plot with explicit path
                plot_path = os.path.join(exp_dir, 'comparison.png')
                print(f"\nSaving plot to: {plot_path}")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved successfully")
                
            except Exception as e:
                print(f"Error during plotting: {str(e)}")
                import traceback
                traceback.print_exc()
            
            finally:
                plt.close('all')  # Always close all figures
            
            # Log final results to wandb
            for acq_type, data in results.items():
                wandb.log({
                    f"final/{acq_type}/best_value": data["best_values"][-1],
                    f"final/{acq_type}/total_cost": data["total_cost"],
                    f"final/{acq_type}/n_iterations": len(data["iterations"]),
                })

            # Log the comparison plot to wandb
            wandb.log({"comparison_plot": wandb.Image(os.path.join(exp_dir, 'comparison.png'))})

            sweep_metadata["completed_configs"] += 1
            
        except Exception as e:
            wandb.log({"error": str(e)})
            print(f"Error in configuration {config_idx}: {str(e)}")
            sweep_metadata["failed_configs"].append({
                "config_idx": config_idx,
                "metric": METRIC_NAMES[config['params']['metric_index']],
                "params": convert_to_serializable(config["params"]),
                "error": str(e)
            })
            import traceback
            traceback.print_exc()
        
        finally:
            # Close the wandb run for this configuration
            wandb.finish()
    
    # Save sweep metadata
    with open(os.path.join(base_results_dir, 'sweep_metadata.json'), 'w') as f:
        json.dump(convert_to_serializable(sweep_metadata), f, indent=2)
    
    print(f"\nSweep completed. Results saved in: {base_results_dir}")
    print(f"Completed configurations: {sweep_metadata['completed_configs']}/{len(configs)}")
    if sweep_metadata["failed_configs"]:
        print(f"Failed configurations: {len(sweep_metadata['failed_configs'])}")

if __name__ == "__main__":
    run_sweep() 