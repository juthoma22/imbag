import pandas as pd
from PIL import Image
import datasets
from datasets import Dataset
from datasets import load_dataset
import os
from tqdm import tqdm


def load_data(csv_file, image_folder_path):
    # Data path
    path = '/home/data_shares/geocv/'

    # Load the CSV file
    metadata = pd.read_csv(f"{path}{csv_file}")

    # Prepare the data
    data = []
    for index, row in tqdm(metadata.iterrows()):
        image_id = row['Index']
        image_path = f'{path}{image_folder_path}/google_{image_id}.jpg'

        try:
            # Load the image with context manager to ensure it is closed after loading
            with Image.open(image_path) as image:
                # Process the image as needed (resize, normalize, etc.)

                # Add the image and its metadata to the dataset
                data.append({
                    'image': image.copy(),  # Create a copy of the image
                    'metadata': row.to_dict()  # All the metadata associated with the image
                })

        except Exception as e:
            print(f"Error loading image {image_id}: {e}")
            pass

    # Create a Hugging Face dataset
    hf_dataset = Dataset.from_pandas(pd.DataFrame(data))

    return hf_dataset


def load_google_data(argument):
    path = f"/home/data_shares/geocv/preprocessed_{argument}_dataset.hf"
    if os.path.exists(path):
        hf_dataset = datasets.load_from_disk(path)
        return hf_dataset
    else:
        print("No dataset found")
    # csv_file = 'google_image_metadata_with_climate_zones.csv'
    # image_folder_path = 'google_images/singles'

    # hf_dataset = load_data(csv_file, image_folder_path)
    
    # hf_dataset.save_to_disk('google_dataset.hf')


def load_mapillary_data():

    csv_file = 'mapillary_image_metadata_with_climate.csv'
    image_folder_path = 'mapillary_images'

    hf_dataset = load_data(csv_file, image_folder_path)

    hf_dataset.save_to_disk('mapillary_dataset.hf')

    return hf_dataset
