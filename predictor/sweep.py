import wandb
import yaml
import argparse
from train import train

def main(is_mini_run=False):
    # Choose which config file to use
    config_file = "config.yaml"
    
    # Load sweep config
    with open(config_file, "r") as f:
        sweep_config = yaml.safe_load(f)
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project="datarecipe")
    

    count = 6 if is_mini_run else 20
    wandb.agent(sweep_id, function=train, count=count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true", help="Run a mini sweep for testing")
    args = parser.parse_args()
    
    main(is_mini_run=args.mini)