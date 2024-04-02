from torch import nn
from transformers import CLIPModel, CLIPImageProcessor
from load_data import load_google_data
import geopandas as gpd
import pandas as pd
from shapely import wkt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import torch
import wandb

def load_embeddings(parquet_path):
    # Load embeddings and IDs from a Parquet file
    embeddings_df = pd.read_parquet(parquet_path)

    # Assuming the embeddings are stored in columns named 'dim0', 'dim1', ..., and IDs are in the 'id' column
    embeddings = embeddings_df.loc[:, embeddings_df.columns != 'id'].values
    ids = embeddings_df['id'].values

    return embeddings, ids

class EmbeddingsDataset(Dataset):
    def __init__(self, dataset, embeddings, ids):
        self.dataset = dataset
        self.embeddings = embeddings
        self.ids = ids
        self.id_to_index = {id_: i for i, id_ in enumerate(ids)}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data_item = self.dataset[idx]
        embedding_index = self.id_to_index[data_item["__index_level_0__"]]
        embedding = self.embeddings[embedding_index]
        
        return {"embedding": embedding, "Geocell": data_item["Geocell"], "id": data_item["__index_level_0__"]}


class FCNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes, dropout):
        super(FCNNClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size2)
        self.output_layer = nn.Linear(hidden_size2, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.output_layer(out)
        return out


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate):
        super(MLPClassifier, self).__init__()
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(), nn.Dropout(dropout_rate)]

        # Adding hidden layers dynamically based on the hidden_sizes list
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        # ModuleList registers all the layers properly
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ImbagClassifier:
    def __init__(self, dataset_path='/home/data_shares/geocv/geocells.csv', dropout=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id_to_latlon = self.load_geocells(path=dataset_path)
        latlon_list = [[value['lat'], value['lon']] for value in self.id_to_latlon.values()]
        self.all_latlons_tensor = torch.tensor(latlon_list, dtype=torch.float32, device=self.device)

        self.classifier = FCNNClassifier(768, 876, 964, 1093, dropout).to(self.device)
        # self.classifier = MLPClassifier(768, [876, 964], 1093, dropout).to(self.device)

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

    def train_classifier(self, embeddings_path, batch_size=16, num_epochs=10, lr=1e-3):
        config = wandb.config
        embeddings, ids = load_embeddings(embeddings_path)
        embeddings = embeddings.astype(np.float32)  # Convert embeddings to float32
        # Load the original dataset
        dataset = load_google_data("imbag_clip_dataset.hf")
        train_dataset = dataset['train']
        eval_dataset = dataset['validation']

        # Wrap with EmbeddingsDataset
        train_dataset = EmbeddingsDataset(train_dataset, embeddings, ids)
        eval_dataset = EmbeddingsDataset(eval_dataset, embeddings, ids)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=lr)
    
        for epoch in range(num_epochs):
            self.classifier.train()
            train_epoch_loss = 0.0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
            for batch in progress_bar:
                embedding = batch["embedding"].to(self.device)
                labels = batch["Geocell"].to(self.device)
                outputs = self.classifier(embedding)
                loss = self.haversine_loss(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                wandb.log({"train_loss": loss.item()})
            avg_train_loss = train_epoch_loss / len(train_loader)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss}')
            wandb.log({"avg_train_loss": avg_train_loss})
            # Evaluation
            self.classifier.eval()
            with torch.no_grad():
                eval_epoch_loss = 0.0
                
                progress_bar = tqdm(eval_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
                for batch in progress_bar:
                    embedding = batch["embedding"].to(self.device)
                    labels = batch["Geocell"].to(self.device)
                    idxs = batch["id"]
                    outputs = self.classifier(embedding)
                    loss = self.haversine_loss(outputs, labels)
                    eval_epoch_loss += loss.item()
                    wandb.log({"eval_loss": loss.item()})
                    
                print(f'Epoch [{epoch+1}/{num_epochs}], Evaluation Loss: {eval_epoch_loss / len(eval_loader)}')
                wandb.log({"avg_eval_loss": eval_epoch_loss/len(eval_loader)})

    def save_model(self, path):
        self.classifier.to('cpu')
        torch.save(self.classifier.state_dict(), path)

def main():
    wandb.init()
    config = wandb.config  # Access the config object
    dataset_path = '/home/data_shares/geocv/geocells.csv'
    embeddings_path = 'notebooks/zesty-forest-48_1_embeddings_with_ids.parquet'

    classifier = ImbagClassifier(dataset_path=dataset_path, dropout=config.dropout)

    classifier.train_classifier(embeddings_path=embeddings_path, batch_size=config.batch_size, num_epochs=config.epochs, lr=config.learning_rate)
    classifier.save_model(f"models/cls_{wandb.run.name}.pth")
    print("Training complete.")

def sweep():
    sweep_config = {
    'method': 'bayes',
    'metric': {
            'name': 'avg_eval_loss',
            'goal': 'minimize'   
        },
    'parameters': {
        'learning_rate': {
            'min': 1e-9,
            'max': 1e-5
            },
        'epochs': {
            'values': [3, 4, 5]
            },
        'batch_size': {
            'values': [16, 32, 64, 128, 265, 512]
            },
        'dropout': {
            'values': [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, project=f"ImbagClassifier")
    return sweep_id

if __name__ == "__main__":
    sweep_id = sweep()
    
    wandb.agent(sweep_id, function=main)