import collections
import pandas as pd
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from GeoClassifier import GeographicalClassifier
import torch

class ImbagClassifier:

    def __init__(self):
        
        model_name = "openai/clip-vit-large-patch14-336"
        self.embeddings_df = pd.read_parquet('data/zesty-forest-48_1_embeddings_with_ids.parquet')
        self.metadata = pd.read_csv('data/imbag_metadata.csv')
        self.device = torch.device("cpu")
        self.classifier = torch.load('data/cls_solar-sweep-285.pth')
        self.classifier.to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.load_state_dict(torch.load("data/zesty-forest-48_1.pth"))
        self.model.to(self.device)
        self.model.eval()

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def find_most_similar(self, target_embedding: np.ndarray, dataset_embeddings: np.ndarray) -> np.ndarray:
        similarities = np.array([self.cosine_similarity(target_embedding, emb) for emb in dataset_embeddings]).squeeze()
        sorted_indices = np.argsort(-similarities)
        return sorted_indices, similarities[sorted_indices]

    def get_closest_image(self, embedded_input_img, embeddings_df):
        embeddings_array = np.array(embeddings_df.drop(columns='id'))
        sorted_indices, sorted_similarities = self.find_most_similar(embedded_input_img, embeddings_array)
        lon, lat = self.metadata.iloc[int(embeddings_df.iloc[sorted_indices[0]].id)][['Longitude', 'Latitude']]
        return lon, lat, sorted_similarities[0]


    def embed_image(self, input_img):
        inputs = self.processor(images=input_img, return_tensors="pt")
        image_tensor = inputs["pixel_values"].to(self.device)
        with torch.no_grad():
            outputs = self.model.get_image_features(image_tensor)
        return outputs

    def make_prediction(self, image_paths, mean_embedding=False):

        selected_geocells_all = []
        selected_values_all = []
        embedded_imgs_all = []

        for image_path in image_paths:
            selected_geocells = []
            selected_values = []
            embedded_imgs = []
            input_img = Image.open(image_path)
            embedded_input_img = self.embed_image(input_img)
            embedded_imgs.append(embedded_input_img)

            output = self.classifier(embedded_input_img)
            values, indices = torch.softmax(output, dim=1).sort(descending=True)

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

            print("Selected geocells:", selected_geocells)
            print("Selected values:", selected_values)

            selected_geocells_all.extend(selected_geocells)
            selected_values_all.extend(selected_values)
            embedded_imgs_all.extend(embedded_imgs)

        geocell_dict = {}
        for geocell, value in zip(selected_geocells_all, selected_values_all):
            geocell_dict[geocell] = geocell_dict.get(geocell, 0) + value

        print("Geocell dict:", geocell_dict)
        
        high_values_geocells = [max(geocell_dict, key=geocell_dict.get)]

        counter = 0
        geocell_dict = dict(sorted(geocell_dict.items(), key=lambda item: item[1], reverse=True))
        for geocell, value in geocell_dict.items():
            print(geocell, value, high_values_geocells[-1], geocell_dict[high_values_geocells[-1]])
            if value >= 0.80 * geocell_dict[high_values_geocells[-1]]:
                if geocell not in high_values_geocells:
                    high_values_geocells.append(geocell)
                    counter += 1
            else:
                break
            if counter == 5:
                break

        print("Selected geocells:", high_values_geocells)

        embeddings_df = self.embeddings_df.copy()

        embeddings_subset = embeddings_df.merge(self.metadata[['Geocell']].reset_index(names='id'), on='id')

        embeddings_subset = embeddings_subset[embeddings_subset['Geocell'].isin(high_values_geocells)].drop(columns='Geocell')

        if embeddings_subset.empty:
            print("No matching geocells found.")
            return

        if mean_embedding == True:

            # take mean of embeddings_imgs_all
            embedded_input_img = torch.stack(embedded_imgs_all).mean(dim=0)

            result = self.get_closest_image(embedded_input_img, embeddings_subset)

            lon, lat = result[:2]

        else:
            results = []
            for embedded_input_img in embedded_imgs_all:

                result = self.get_closest_image(embedded_input_img, embeddings_subset)
                results.append(result)
                print(result)
            
            best_result = max(results, key=lambda x: x[2])
            lon, lat = best_result[:2]
    
        print(f"Predicted latitude: {lat}, longitude: {lon}")
        guess = self.metadata[(self.metadata['Latitude'] == lat) & (self.metadata['Longitude'] == lon)].iloc[0]

        return guess['Country'], guess['Latitude'], guess['Longitude'], guess["Geocell"], selected_values_all
    
if __name__ == '__main__':
    
    classifier = ImbagClassifier()
    classifier.make_prediction('imgs/comte.png')