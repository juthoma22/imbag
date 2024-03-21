from torch.utils.data import DataLoader
import wandb
from load_data import load_google_data

import sys

from ClipFineTuner import CLIPFineTuner

# take argument from command line
argument = sys.argv[1]
print(f"Argument: {argument}")
if argument not in ["climate_zone", "Country"]:
    print("Invalid argument")
    sys.exit()
if argument == "climate_zone":
    argument = "Climate Zone"
print(f"Argument: {argument}")


def prepare_dataloader(dataset, batch_size=8):
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'pixel_values'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

def train_model():
    wandb.init()
    config = wandb.config 
    
    print("Initializing model...")
    finetuner = CLIPFineTuner(
        model_name="openai/clip-vit-large-patch14-336",
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        argument=argument
    )

    print("Loading dataset...")
    dataset = load_google_data(argument)
    train_dataset, validation_dataset = dataset['train'], dataset['validation']

    print("Loading dataloader...")
    train_loader, validation_loader = prepare_dataloader(train_dataset, batch_size=config.batch_size), prepare_dataloader(validation_dataset, batch_size=config.batch_size)

    print("Training model...")
    finetuner.train(train_loader, validation_loader)
    finetuner.save_model()

def sweep():
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 1e-9, "max": 1e-5},
            "batch_size": {"values": [8, 16, 32]},
            "epochs": {"values": [2, 3, 4]}
        },
        "early_terminate": {
            "type": "hyperband",
            "s": 2,  # defines the reduction factor
            "eta": 3,  # defines the proportion of configurations that are discarded in each iteration
            "min_iter": 2,  # maximum number of iterations to run
            "max_iter": 6
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=f"{argument}CLip")
    return sweep_id

if __name__ == "__main__":
    sweep_id = sweep()
    wandb.agent(sweep_id, function=train_model)
