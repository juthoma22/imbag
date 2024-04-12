import pandas as pd
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor
import torch
import argparse
from math import radians, cos, sin, asin, sqrt, exp

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_most_similar(target_embedding: np.ndarray, dataset_embeddings: np.ndarray) -> np.ndarray:
    similarities = np.array([cosine_similarity(target_embedding, emb) for emb in dataset_embeddings])
    sorted_indices = np.argsort(-similarities)
    return sorted_indices, similarities[sorted_indices]

def get_closest_image(embedded_input_img, embeddings_df, metadata):
    embeddings_array = np.array(embeddings_df.drop(columns='id'))
    sorted_indices, sorted_similarities = find_most_similar(embedded_input_img, embeddings_array)
    lon, lat = metadata.iloc[int(embeddings_df.iloc[sorted_indices[0]].id)][['Longitude', 'Latitude']]
    return lon, lat


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
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
    Calculate the score based on the distance using the provided formula.
    """
    return 5000 * exp(-distance / 1492.7)

def embed_image(input_img, processor, model, device):
    inputs = processor(images=input_img, return_tensors="pt")
    image_tensor = inputs["pixel_values"].to(device)
    with torch.no_grad():
        outputs = model.get_image_features(image_tensor)
    return outputs.cpu().numpy()

def main(args, model_name = "openai/clip-vit-large-patch14-336",
         model_path = "/home/data_shares/geocv/models/zesty-forest-48_1.pth",
         input_img_path = "/home/data_shares/geocv/test_img.png",
         embeddings_path = '/home/data_shares/geocv/zesty-forest-48_1_embeddings_with_ids.parquet',
         metadata_path = '/home/data_shares/geocv/imbag_metadata.csv'
         ):
    
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    embeddings_df = pd.read_parquet(embeddings_path)
    metadata_df = pd.read_csv(metadata_path)

    geocells = [int(geocell) for geocell in args.geocells] 

    embeddings_subset = embeddings_df.merge(metadata_df[['Geocell']].reset_index(names='id'), on='id')
    embeddings_subset = embeddings_subset[embeddings_subset['Geocell'].isin(geocells)].drop(columns='Geocell')

    if embeddings_subset.empty:
        print("No matching geocells found.")
        return

    input_img = Image.open(input_img_path)
    embedded_input_img = embed_image(input_img, processor, model, device)
    print(embedded_input_img.shape)
    print(embeddings_subset.shape)
    print(embeddings_subset)

    lon, lat = get_closest_image(embedded_input_img, embeddings_subset, metadata_df)
    distance = haversine(lon, lat, float(args.lon), float(args.lat))
    
    score = calculate_score(distance)
    print(f'Latitude: {lat}, Longitude: {lon}')
    print(f'Google Maps Link: https://www.google.com/maps/@{lat},{lon},17z?entry=ttu')
    print(f'Score: {score}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find the closest image based on CLIP embeddings and score it.')

    parser.add_argument('--geocells', nargs='+', required=True, help='List of geocells of interest')
    parser.add_argument('--input_img_path', required=False, help='Path to the input image file')
    parser.add_argument('--lon', type=float, required=True, help='Reference longitude for scoring')
    parser.add_argument('--lat', type=float, required=True, help='Reference latitude for scoring')

    args = parser.parse_args()

    main(args)
