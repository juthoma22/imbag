import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.functional import to_tensor
from transformers import CLIPProcessor, CLIPModel
from datasets import Dataset
from tqdm import tqdm
from torchvision import transforms
import wandb
from load_data import load_google_data
import numpy as np
import sys

climate_zone_descriptions = {
    "Af": "Tropical rainforest climate",
    "Am": "Tropical monsoon climate",
    "Aw": "Tropical savanna climate (wet-dry)",
    "BWh": "Hot desert climate",
    "BWk": "Cold desert climate",
    "BSh": "Hot semi-arid climate",
    "BSk": "Cold semi-arid climate",
    "Csa": "Hot-summer Mediterranean climate",
    "Csb": "Warm-summer Mediterranean climate",
    "Csc": "Cold-summer Mediterranean climate",
    "Cwa": "Monsoon-influenced humid subtropical climate",
    "Cwb": "Subtropical highland climate with dry winters",
    "Cwc": "Cold subtropical highland climate with dry winters",
    "Cfa": "Humid subtropical climate",
    "Cfb": "Oceanic climate",
    "Cfc": "Subpolar oceanic climate",
    "Dsa": "Hot-summer Mediterranean continental climate",
    "Dsb": "Warm-summer Mediterranean continental climate",
    "Dsc": "Cold-summer Mediterranean continental climate",
    "Dsd": "Cold-summer Mediterranean continental climate with extremely cold winters",
    "Dwa": "Monsoon-influenced hot-summer humid continental climate",
    "Dwb": "Monsoon-influenced warm-summer humid continental climate",
    "Dwc": "Monsoon-influenced subarctic climate",
    "Dwd": "Monsoon-influenced subarctic climate with extremely cold winters",
    "Dfa": "Hot-summer humid continental climate",
    "Dfb": "Warm-summer humid continental climate",
    "Dfc": "Subarctic climate with cool summers",
    "Dfd": "Subarctic climate with extremely cold winters",
    "ET": "Tundra climate",
    "EF": "Ice cap climate"
}

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# take argument from command line
argument = sys.argv[1]
print(f"Argument: {argument}")
if argument not in ["climate_zone", "Country"]:
    print("Invalid argument")
    sys.exit()
if argument == "climate_zone":
    argument = "Climate Zone"
print(f"Argument: {argument}")

def process_data(examples):
    captions = []
    images = []
    for arg, image in zip(examples[argument], examples["image"]):
        images.append(image)
        if argument == "Climate Zone":
            caption = "this image was taken in " + climate_zone_descriptions.get(arg).lower()
        if argument == "Country":
            caption = "this image was taken in " + arg
        else:
            print("Invalid argument")
        captions.append(caption)

    return processor(text=captions, images=images, return_tensors="pt", padding="max_length", max_length=32, truncation=True)

def prepare_dataloader(dataset, batch_size=8):
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'pixel_values'])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

class CLIPFineTuner:
    def __init__(self, model_name, batch_size=8, learning_rate=5e-5, epochs=4):
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        else:
            self.device = torch.device("cpu")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch, return_loss=True)
                loss = outputs.loss
                total_loss += loss.item()

                logits_per_image = outputs.logits_per_image
                predictions = logits_per_image.argmax(dim=-1)
                correct = (predictions == torch.arange(len(predictions), device=self.device)).sum()
                total_correct += correct.item()
                total_samples += len(predictions)

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def train(self, train_loader, validation_loader):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch, return_loss=True)
                loss = outputs.loss
                
                total_train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss, val_accuracy = self.evaluate(validation_loader)

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy
            })
            print(f"Epoch {epoch+1} - Avg Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    def save_model(self):
        path = f'models/{argument}clip_{self.learning_rate}_{self.batch_size}_{self.epochs}.pth'
        self.model.to('cpu')
        torch.save(self.model.state_dict(), path)

def load_and_preprocess_data():
    dataset = load_google_data()
    train_dataset, validation_dataset = dataset['train'], dataset['validation']
    workers = torch.cuda.device_count()
    preprocessed_train_dataset = train_dataset.map(process_data, batched=True, batch_size=1000, num_proc=workers, remove_columns=["image", "Climate Zone", "Country"])
    preprocessed_validation_dataset = validation_dataset.map(process_data, batched=True, batch_size=1000, num_proc=workers, remove_columns=["image", "Climate Zone", "Country"])
    return preprocessed_train_dataset, preprocessed_validation_dataset

def train_model():
    wandb.init()
    config = wandb.config 

    train_loader, validation_loader = prepare_dataloader(train_dataset, batch_size=config.batch_size), prepare_dataloader(validation_dataset, batch_size=config.batch_size)

    finetuner = CLIPFineTuner(
        model_name="openai/clip-vit-base-patch32",
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        epochs=config.epochs
    )

    finetuner.train(train_loader, validation_loader)
    finetuner.save_model()

def sweep():
    sweep_config = {
        "method": "bayes",
        "metric": {"name": "val_loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 1e-9, "max": 1e-5},
            "batch_size": {"values": [32, 64, 128]},
            "epochs": {"values": [8, 12, 16]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=f"{argument}CLip")
    return sweep_id

if __name__ == "__main__":
    train_dataset, validation_dataset = load_and_preprocess_data()
    sweep_id = sweep()
    wandb.agent(sweep_id, function=train_model)
