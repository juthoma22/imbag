from ClipFineTuner import CLIPFineTuner
import transformers
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPImageProcessor, ViTImageProcessor
from torch.utils.data import DataLoader
import wandb
from load_data import load_google_data
from PIL import Image
import sys
import copy
import torch
import random
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import geopandas as gpd
import pandas as pd
from shapely import wkt
import numpy as np

def calculate_accuracy(logits, labels):
    """
    Calculate accuracy of predictions.

    Parameters:
    - logits (torch.Tensor): The logits output by the model. Shape: [batch_size, num_classes].
    - labels (torch.Tensor): The true labels. Shape: [batch_size].

    Returns:
    - accuracy (float): The accuracy of predictions as a percentage.
    """
    # Convert logits to predicted class indices
    preds = torch.argmax(logits, dim=1)
    # Calculate the number of correct predictions
    correct_predictions = torch.sum(preds == labels)
    
    # Calculate accuracy
    accuracy = (correct_predictions / labels.size(0)) * 100.0  # Convert to percentage
    
    return accuracy.item()


class ViTClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ViTClassifierHead, self).__init__()
        # Replace the classifier with one that matches your number of classes
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, features):
        return self.classifier(features.pooler_output)


class ImbagViT():
    def __init__(self, model_path, batch_size=8, learning_rate=1e-5, epochs=5):
        """
        Take dataset configuration
        Load dataset in that configuration
        Initialize dataloader
        Initialize a CLIPFineTuner
        Train CLIPFineTuner on the dataset

        Args:
        mode (str): "country_first" for Country -> Climate -> Geocell, "climate_first" for Climate -> Country -> Geocell, "separate" for mixed in separate captions, "combined" for a single combined caption
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        print(f"Hyperparameters: {self.batch_size}, {self.learning_rate}, {self.epochs}")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
        model.load_state_dict(torch.load(model_path))
        vision_model = model.vision_model
        vision_model.classifier_head = ViTClassifierHead(vision_model.config.hidden_size, 1093)
        self.model = vision_model
        # make whole model trainable
        for param in self.model.parameters():
            param.requires_grad = True
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model = self.model.to(self.device)
        else:
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
        # wandb.init() # TODO: add project name

        
        self.id_to_latlon = self.load_geocells('/home/data_shares/geocv/geocells.csv')
        latlon_list = [[value['lat'], value['lon']] for value in self.id_to_latlon.values()]
        self.all_latlons_tensor = torch.tensor(latlon_list, dtype=torch.float32, device=self.device)
        self.loss_fn = self.haversine_loss
        self.loss_fn = nn.CrossEntropyLoss()

    def process_data(self, examples):
        """
        Transform the input image to the feature vectors to be used by the model.

        Args:
            example_batch: A batch of input images.

        Returns:
            The transformed input features with labels.
        """
        images = []
        for file_path in examples["File Path"]:
                img = Image.open(file_path)
                images.append(img)

        inputs = self.image_processor(images, return_tensors='pt')

        # include the labels
        inputs['labels'] = examples['Geocell']
        return inputs
    

    def preprocess(self):
        """
        Using CLIP Processor to preprocess the dataset

        Returns:
        dataset: dataset object
        """
        dataset = load_google_data("imbag_clip_dataset.hf")
        self.train_dataset, self.validation_dataset = dataset['train'], dataset['validation']
        # self.train_dataset = self.train_dataset.select([random.randint(0, len(self.train_dataset)) for _ in range(2)])
        # self.validation_dataset = self.validation_dataset.select([random.randint(0, len(self.train_dataset)) for _ in range(2)])

        self.train_dataset = self.train_dataset.map(lambda x: self.process_data(x), batched=True, batch_size=500, remove_columns=["File Path", "Climate Zone", "Country", "Geocell"])
        self.validation_dataset = self.validation_dataset.map(lambda x: self.process_data(x), batched=True, batch_size=500, remove_columns=["File Path", "Climate Zone", "Country", "Geocell"])


    def prepare_dataloader(self):
        """
        Prepare dataloader for training
        """
        batch_size = self.batch_size

        self.train_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
        self.validation_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)


    def load_geocells(self, path):
        df = pd.read_csv(path)

        df["geometry"] = df["geometry"].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df)

        # Calculate centroids
        gdf['centroid'] = gdf.centroid

        # Extract latitudes and longitudes
        gdf['lat'] = gdf.centroid.y
        gdf['lon'] = gdf.centroid.x

        # Create a dictionary for quick ID to lat, lon lookup
        id_to_latlon = gdf[['lat', 'lon']].to_dict(orient='index')
        
        return id_to_latlon


    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_items = 0

        with torch.no_grad():  # No need to track gradients during evaluation
            for batch in tqdm(self.validation_loader, desc=f"Evaluating:.."):
                # print(batch)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch["labels"]
                images = batch["pixel_values"]
                
                # Forward pass
                features = self.model(images)
                logits = self.model.classifier_head(features)
                loss = self.loss_fn(logits, labels)
                
                # Accumulate loss and accuracy
                total_loss += loss.item() * images.size(0)  # Multiply by batch size to get total loss for the batch
                total_accuracy += calculate_accuracy(logits, labels) * images.size(0)  # Use the previously defined function
                total_items += images.size(0)
        
        # Calculate average loss and accuracy
        average_loss = total_loss / total_items
        average_accuracy = total_accuracy / total_items

        return average_loss, average_accuracy


    def calculate_haversine_distance(self, latlon1, latlon2):
        """
        Calculate the Haversine distance between two sets of points (latlon1 and latlon2).
        Each parameter is a tensor of shape [batch_size, 2], where each row contains latitude and longitude values.

        Parameters:
        - latlon1: Tensor of shape [batch_size, 2]. Represents the first set of points.
        - latlon2: Tensor of shape [batch_size, 2]. Represents the second set of points.

        Returns:
        - distances: Tensor of shape [batch_size]. The Haversine distance between corresponding points in latlon1 and latlon2.
        """
        R = 6371.0  # Radius of Earth in kilometers
        # Convert degrees to radians
        lat1, lon1 = torch.deg2rad(latlon1[:, 0]), torch.deg2rad(latlon1[:, 1])
        lat2, lon2 = torch.deg2rad(latlon2[:, 0]), torch.deg2rad(latlon2[:, 1])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        distance = R * c

        return distance


    def haversine_loss(self, outputs, target_ids):
        # Convert logits to probabilities
        probabilities = F.softmax(outputs, dim=1)

        # Get target lat, lon for comparison
        # Convert target_ids to lat, lon pairs, then to a tensor
        target_latlons_list = [[self.id_to_latlon[id.item()]['lat'], self.id_to_latlon[id.item()]['lon']] for id in target_ids]
        target_latlons_array = np.array(target_latlons_list, dtype=np.float32)  # Convert list to numpy array with float32 type

        # Create a PyTorch tensor from the numpy array and move it to the correct device
        target_latlons = torch.tensor(target_latlons_array, device=outputs.device)  # Shape [batch_size, 2]

        # Calculate distances from all classes to the target, shape [batch_size, num_classes]
        all_distances = torch.stack([self.calculate_haversine_distance(self.all_latlons_tensor, target_latlon.repeat(self.all_latlons_tensor.shape[0], 1)) for target_latlon in target_latlons])

        # Weight distances by probabilities and sum across classes for each item in batch
        weighted_distances = torch.sum(all_distances * probabilities, dim=1)

        # Aggregate over the batch
        return torch.mean(weighted_distances)


    def train(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0
            
            total_accuracy = 0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                # print(batch)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch["labels"]
                images = batch["pixel_values"]
                features = self.model(images)

                logits = self.model.classifier_head(features)

                loss = self.loss_fn(logits, labels)
                total_train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_accuracy = calculate_accuracy(logits, labels)
                total_accuracy += batch_accuracy

                wandb.log({
                    "batch_accuracy": batch_accuracy,
                    "batch_loss": loss.item()
                })

            average_accuracy = total_accuracy / len(self.train_loader)
            avg_train_loss = total_train_loss / len(self.train_loader)
            val_loss, val_accuracy = self.evaluate()

            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "train_accuracy": average_accuracy,
                "val_accuracy": val_accuracy,
                "val_loss": val_loss,
            })

            print(f"Epoch {epoch+1}, Training Accuracy: {average_accuracy:.4f}")
            print(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss:.4f}")
            print(f"Epoch {epoch+1}, Validation Accuracy Geocell: {val_accuracy:.4f}")
            print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

            # run id from wandb
            run_id = wandb.run.name
            path = f'models/vit_{run_id}_{epoch}.pth'
            self.save_model(path)
            self.model.to(self.device)
    
    def save_model(self, path):
        self.model.to('cpu')
        torch.save(self.model.state_dict(), path)


if __name__ == "__main__":
    epochs = int(sys.argv[1])
    print(f"Running ViT Classification Traingin...")
    imbag_clip = ImbagViT(model_path="/home/jthr/repos/imbag/models/zesty-forest-48_1.pth",epochs=epochs)
    imbag_clip.preprocess()
    imbag_clip.prepare_dataloader()
    imbag_clip.train()