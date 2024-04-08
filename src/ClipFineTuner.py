import torch
import wandb
from tqdm import tqdm
import pandas as pd
import copy
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader

import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
import datasets
import sys
from tqdm import tqdm
from PIL import Image
import copy
import pandas as pd
import random


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

def calculate_accuracy(true_labels, predicted_labels):
    """
    Calculates the accuracy of a classifier given the true labels and predicted labels.

    Args:
    true_labels (list): List of true labels.
    predicted_labels (list): List of predicted labels.

    Returns:
    float: Accuracy of the classifier.
    """
    correct = 0
    total = len(true_labels)
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label == predicted_label:
            correct += 1
    accuracy = correct / total
    return accuracy


class CLIPFineTuner:
    def __init__(self, model_name="openai/clip-vit-large-patch14-336", batch_size=8, learning_rate=5e-6, epochs=3, argument="all"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.argument = argument
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

        print("Loading model...")
        # Set up model and optimizer
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        else:
            self.device = torch.device("cpu")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)


    def evaluate(self, dataset, argument):
        self.model.eval()
        true_labels = []
        predicted_labels = []

        data = copy.deepcopy(dataset)
        data.set_format(columns=['File Path', 'Climate Zone', 'Country', 'Geocell'])

        countries = list(pd.Series(dataset["Country"]).unique())
        labels_country = [f"this image was taken in {c}" for c in countries]
        climates = [c for c in climate_zone_descriptions.keys()]
        labels_climate = [f"this image was taken in {c.lower()}" for c in climate_zone_descriptions.values()]
        geocells = list(pd.Series(dataset["Geocell"]).unique())
        labels_geocell = [f"this image was taken within geocell {c}" for c in geocells]

        # print(argument)
        # print(f"labels climate: {labels_climate}")
        dataloader = DataLoader(data, batch_size=8)
        for batch in tqdm(dataloader, desc="Evaluating:.."):
            imgs = []
            captions = []
            for arg, image in zip(batch[argument], batch["File Path"]):
                img = Image.open(image)
                imgs.append(img)
                if argument == "Country":
                    caption = "this image was taken in " + arg
                elif argument == "Climate Zone":
                    caption = "this image was taken in " + climate_zone_descriptions.get(arg).lower()
                elif argument == "Geocell":
                    caption = "this image was taken within geocell " + str(arg)

                captions.append(caption)


            if argument == "Country":
                inputs = self.processor(text=labels_country, images=imgs, return_tensors="pt", padding=True)
            elif argument == "Climate Zone":
                inputs = self.processor(text=labels_climate, images=imgs, return_tensors="pt", padding=True)
            elif argument == "Geocell":
                inputs = self.processor(text=labels_geocell, images=imgs, return_tensors="pt", padding=True)

            true_labels.extend(captions)
            # if argument == "Climate Zone":
                # print(f"True labels: {captions}")

            # Move inputs to GPU if available
            for k, v in inputs.items():
                inputs[k] = v.to(self.device)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get predicted labels
            logits_per_image = outputs.logits_per_image


            if argument == "Country":
                predicted_labels.extend([labels_country[i.argmax().item()] for i in logits_per_image])
            elif argument == "Climate Zone":
                predicted_labels.extend([labels_climate[i.argmax().item()] for i in logits_per_image])
            elif argument == "Geocell":
                predicted_labels.extend([labels_geocell[i.argmax().item()] for i in logits_per_image])
        
        # if argument == "Climate Zone":            
        #     print(f"Predicted labels: {predicted_labels}")
        accuracy = calculate_accuracy(true_labels, predicted_labels)

        return accuracy


    def train(self, train_loader, validation_dataset):
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
            val_accuracy_country = self.evaluate(validation_dataset, "Country")
            val_accuracy_climate = self.evaluate(validation_dataset, "Climate Zone")

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_accuracy_country": val_accuracy_country,
                "val_accuracy_climate": val_accuracy_climate,
            })
            print(f"Epoch {epoch+1}, Validation Accuracy Country: {val_accuracy_country:.4f}, Validation Accuracy Climate: {val_accuracy_climate:.4f}")

            # run id from wandb
            run_id = wandb.run.name
            path = f'models/{run_id}_{epoch}.pth'
            self.save_model(path)
            self.model.to(self.device)

    def save_model(self, path):
        self.model.to('cpu')
        torch.save(self.model.state_dict(), path)