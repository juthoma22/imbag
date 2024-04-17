import pandas as pd
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ImbagTrainClassifier import GeographicalClassifier
import torch
from math import radians, cos, sin, asin, sqrt, exp

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_most_similar(target_embedding: np.ndarray, dataset_embeddings: np.ndarray) -> np.ndarray:
    similarities = np.array([cosine_similarity(target_embedding, emb) for emb in dataset_embeddings]).squeeze()
    sorted_indices = np.argsort(-similarities)
    return sorted_indices, similarities[sorted_indices]

def get_closest_image(embedded_input_img, embeddings_df, metadata):
    embeddings_array = np.array(embeddings_df.drop(columns='id'))
    sorted_indices, sorted_similarities = find_most_similar(embedded_input_img, embeddings_array)
    lon, lat = metadata.iloc[int(embeddings_df.iloc[sorted_indices[0]].id)][['Longitude', 'Latitude']]
    return lon, lat


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate haversine distance between two points on the earth.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # Radius of earth in kilometers
    return c * r

def calculate_score(distance):
    """
    Calculate the score based on the distance using the Geoguessr formula.
    """
    return 5000 * exp(-distance / 1492.7)

def embed_image(input_img, processor, model, device):
    inputs = processor(images=input_img, return_tensors="pt")
    image_tensor = inputs["pixel_values"].to(device)
    with torch.no_grad():
        outputs = model.get_image_features(image_tensor)
    return outputs

def main(input_lon, input_lat):
    
    input_img = Image.open(input_img_path)
    embedded_input_img = embed_image(input_img, processor, model, device)


    embeddings_df = pd.read_parquet(embeddings_path)
    metadata_df = pd.read_csv(metadata_path)
    
    top_k_geocells = classifier(embedded_input_img)
    values, indices = torch.softmax(top_k_geocells, dim=1).sort(descending=True)

    selected_geocells = []
    selected_values = []

    for value, idx in zip(values[0], indices[0]):
        try:
            numeric_value = value.item()
            if numeric_value > 0.10:
                selected_geocells.append(int(idx))
                selected_values.append(numeric_value)
        except AttributeError:
            print("Error processing value:", value)

    if sum(selected_values) < 0.50:
        for value, idx in zip(values[0], indices[0]):
            try:
                numeric_value = value.item()
                if numeric_value <= 0.10:
                    selected_geocells.append(int(idx))
                    selected_values.append(numeric_value)
                    if sum(selected_values) >= 0.50:
                        break
            except AttributeError:
                print("Error processing value:", value)


    embeddings_subset = embeddings_df.merge(metadata_df[['Geocell']].reset_index(names='id'), on='id')
    embeddings_subset = embeddings_subset[embeddings_subset['Geocell'].isin(selected_geocells)].drop(columns='Geocell')

    if embeddings_subset.empty:
        print("No matching geocells found.")
        return

    lon, lat = get_closest_image(embedded_input_img, embeddings_subset, metadata_df)
    guess = metadata_df[(metadata_df['Latitude'] == lat) & (metadata_df['Longitude'] == lon)].iloc[0]
    distance = haversine(lon, lat, float(input_lon), float(input_lat))
    
    score = calculate_score(distance)
    print(f"Latitude: {lat}, Longitude: {lon} - {guess['Country']}, {guess['Region']}\n")
    print(f'Google Maps Link: https://www.google.com/maps/@{lat},{lon},17z?entry=ttu')
    print(f'Distance: {round(distance)} km')
    print(f'Score: {round(score)} points\n')

    
    for geocell, value in zip(selected_geocells, selected_values):
        print(f"Geocell: {geocell}\t Country: {metadata_df[metadata_df['Geocell'] == geocell].Country.value_counts().index[0]}\t Probability: {round(value,4)}")
    return score

if __name__ == '__main__':
    model_name = "openai/clip-vit-large-patch14-336"
    model_path = "/home/data_shares/geocv/models/zesty-forest-48_1.pth"
    input_img_path = "/home/data_shares/geocv/polska.png"
    embeddings_path = '/home/data_shares/geocv/zesty-forest-48_1_embeddings_with_ids.parquet'
    metadata_path = '/home/data_shares/geocv/imbag_metadata.csv'
    classifier_path = '/home/data_shares/geocv/models/cls_solar-sweep-285.pth'
    device = torch.device("cpu")
    classifier = torch.load(classifier_path)
    classifier.to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    results = main(19.0117009,51.4775656)
