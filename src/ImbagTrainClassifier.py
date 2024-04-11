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


def calculate_top_k_accuracy(logits, labels, k=1):
    """
    Calculate top-k accuracy of predictions.

    Parameters:
    - logits (torch.Tensor): The logits output by the model. Shape: [batch_size, num_classes].
    - labels (torch.Tensor): The true labels. Shape: [batch_size].
    - k (int): The value of 'k' in top-k accuracy.

    Returns:
    - accuracy (float): The top-k accuracy of predictions as a percentage.
    """
    # Get the top k predictions from the logits
    _, top_k_preds = torch.topk(logits, k, dim=1)
    
    # Check if the true labels are in the top k predictions
    correct_predictions = top_k_preds.eq(labels.view(-1, 1).expand_as(top_k_preds))
    
    # Calculate the number of correct predictions
    correct_predictions = correct_predictions.sum().item()
    
    # Calculate accuracy
    accuracy = (correct_predictions / labels.size(0)) * 100.0  # Convert to percentage
    
    return accuracy


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

class GeographicalClassifier(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dims=[512, 256, 128], num_classes=1093, dropout_rate=0.5):
        super(GeographicalClassifier, self).__init__()
        
        # Dynamically create layers based on the hidden_dims list
        self.layers = nn.ModuleList()
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim  # Set next layer's input size
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        # Apply output layer
        x = self.output_layer(x)
        return x


class ImbagClassifier:
    def __init__(self, dataset_path='/home/data_shares/geocv/geocells.csv', dropout=0.5, hidden_layers=[512, 256, 128], label_smoothing_constant=70):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latlons = self.load_geocells(path=dataset_path)
        self.latlons = self.latlons.to(self.device)
        
        self.label_smoothing_constant = label_smoothing_constant
        self.loss_fn = nn.CrossEntropyLoss()

        self.classifier = GeographicalClassifier(embedding_dim=768, hidden_dims=hidden_layers, num_classes=1093, dropout_rate=dropout).to(self.device)
        # self.classifier = MLPClassifier(768, [876, 964], 1093, dropout).to(self.device)
        # self.classifier = FCNNClassifier(867, 964, 1040, 1093, dropout).to(self.device)

    def load_geocells(self, path):
        df = pd.read_csv(path)

        # Convert WKT column to geometry
        df["geometry"] = df["geometry"].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df)

        # Calculate centroids and extract latitudes and longitudes
        gdf['centroid'] = gdf.centroid
        gdf['lat'] = gdf['centroid'].y
        gdf['lon'] = gdf['centroid'].x

        # Convert DataFrame to PyTorch tensor
        latlons = torch.tensor(gdf[['lat', 'lon']].values, dtype=torch.float32)

        return latlons
    
    def smooth_distances(self, distances):
            """
            Smooths the distances by subtracting the minimum distance within each batch and applying
            exponential smoothing.

            Parameters:
            - distances: A 2D Tensor where each row represents distances from one item
            to multiple geocells.
            - label_smoothing_constant (float): The constant used for smoothing (decay rate in the exponential).

            Returns:
            - Tensor: The smoothed labels.
            """
            label_smoothing_constant = self.label_smoothing_constant
            # Normalize distances by subtracting the minimum distance in each row
            adjusted_distances = distances - distances.min(dim=1, keepdim=True)[0]
            
            # Apply exponential decay to the adjusted distances
            smoothed_labels = torch.exp(-adjusted_distances / label_smoothing_constant)
            
            # Handle NaN and Inf values by replacing them with zero
            smoothed_labels = torch.nan_to_num(smoothed_labels, nan=0.0, posinf=0.0, neginf=0.0)
            
            return smoothed_labels
    
    def calculate_haversine_distance(self, latlon_batch, latlon_geocells):
        """
        Calculate the Haversine distance between a batch of points and a set of geocells.
        
        Parameters:
        - latlon_batch: Tensor of shape [batch_size, 2]. Each row contains latitude and longitude values for a point.
        - latlon_geocells: Tensor of shape [2, number_of_geocells]. Each column contains latitude and longitude values for a geocell.

        Returns:
        - distances: Tensor of shape [batch_size, number_of_geocells]. The Haversine distance from each point in the batch to each geocell.
        """
        R = 6371.0  # Radius of Earth in kilometers
        # Convert degrees to radians
        lat1, lon1 = torch.deg2rad(latlon_batch[:, 0]), torch.deg2rad(latlon_batch[:, 1])
        lat2, lon2 = torch.deg2rad(latlon_geocells[0, :]), torch.deg2rad(latlon_geocells[1, :])

        # Expand lat1, lon1 to match the shape of lat2, lon2 for broadcasting
        lat1 = lat1.unsqueeze(1).expand(-1, lat2.size(0))
        lon1 = lon1.unsqueeze(1).expand(-1, lon2.size(0))

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
        distances = R * c

        return distances

    def train_classifier(self, embeddings_path, batch_size, num_epochs, lr):
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

        optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=lr)
    
        for epoch in range(num_epochs):
            self.classifier.train()
            train_epoch_loss = 0.0
            total_accuracy = 0
            total_top_k_accuracy = 0
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
            for batch in progress_bar:
                embedding = batch["embedding"].to(self.device)
                labels = batch["Geocell"].to(self.device)
                outputs = self.classifier(embedding)
                # pred = torch.softmax(outputs, dim=1)
                latlons_labels = self.latlons[labels]

                distance = self.calculate_haversine_distance(latlons_labels, self.latlons.t())
                smoothed_labels = self.smooth_distances(distance)
                loss = self.loss_fn(outputs, smoothed_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                accuracy = calculate_top_k_accuracy(outputs, labels)
                top_k_accuracy = calculate_top_k_accuracy(outputs, labels, k=5)

                total_accuracy += accuracy
                total_top_k_accuracy += top_k_accuracy

                train_epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            
            avg_train_loss = train_epoch_loss / len(train_loader)
            average_accuracy = total_accuracy / len(train_loader)
            average_top_k_accuracy = total_top_k_accuracy / len(train_loader)
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Accuracy: {average_accuracy}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Training Top-5 Accuracy: {average_top_k_accuracy}')
            wandb.log({"avg_train_loss": avg_train_loss})
            wandb.log({"avg_accuracy": average_accuracy})
            wandb.log({"avg_top_k_accuracy": average_top_k_accuracy})
            # Evaluation
            self.classifier.eval()
            with torch.no_grad():
                eval_epoch_loss = 0.0
                eval_epoch_accuracy = 0.0
                total_top_k_accuracy = 0
                
                progress_bar = tqdm(eval_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
                for batch in progress_bar:
                    embedding = batch["embedding"].to(self.device)
                    labels = batch["Geocell"].to(self.device)
                    outputs = self.classifier(embedding)
                    
                    latlons_labels = self.latlons[labels]
                    distance = self.calculate_haversine_distance(latlons_labels, self.latlons.t())
                    smoothed_labels = self.smooth_distances(distance)

                    loss = self.loss_fn(outputs, smoothed_labels)
                    eval_epoch_loss += loss.item()

                    eval_accuracy = calculate_top_k_accuracy(outputs, labels)
                    top_k_accuracy = calculate_top_k_accuracy(outputs, labels, k=5)
                    eval_epoch_accuracy += eval_accuracy
                    total_top_k_accuracy += top_k_accuracy
                    
                print(f'Epoch [{epoch+1}/{num_epochs}], Evaluation Loss: {eval_epoch_loss / len(eval_loader)}')
                print(f'Epoch [{epoch+1}/{num_epochs}], Evaluation Accuracy: {eval_epoch_accuracy / len(eval_loader)}')
                print(f'Epoch [{epoch+1}/{num_epochs}], Evaluation Top-5 Accuracy: {total_top_k_accuracy / len(eval_loader)}')
                wandb.log({"avg_eval_loss": eval_epoch_loss/len(eval_loader)})
                wandb.log({"avg_eval_accuracy": eval_epoch_accuracy/len(eval_loader)})
                wandb.log({"avg_eval_top_k_accuracy": total_top_k_accuracy/len(eval_loader)})

    def save_model(self, path):
        self.classifier.to('cpu')
        torch.save(self.classifier, path)

def main():
    wandb.init()
    config = wandb.config  # Access the config object
    dataset_path = '/home/data_shares/geocv/geocells.csv'
    embeddings_path = '/home/data_shares/geocv/zesty-forest-48_1_embeddings_with_ids.parquet'
    print(embeddings_path)

    classifier = ImbagClassifier(dataset_path=dataset_path, dropout=config.dropout, hidden_layers=config.hidden_layers)

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
            'min': 1e-10,
            'max': 1e-3
            },
        'epochs': {
            'values': [6, 8, 10, 12, 14, 16, 18, 20]
            },
        'batch_size': {
            'values': [16, 32, 64]
            },
        'dropout': {
            'values': [0.2, 0.3, 0.4, 0.5]
            },
        'hidden_layers': {
            'values': [[840, 960, 1080], [960, 1080], [960]]
            },
        'label_smoothing_constant': {
            'values': [50, 60, 70, 80, 90]
            }
        }
    }
    sweep_id = wandb.sweep(sweep_config, project=f"ImbagClassifierHaversine")
    return sweep_id

if __name__ == "__main__":
    sweep_id = sweep()
    
    wandb.agent(sweep_id, function=main)