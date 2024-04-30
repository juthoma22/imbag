import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
from load_data import load_google_data
from sklearn.preprocessing import OneHotEncoder

# Load the model and processor
model_path = 'graceful-paper-87_5'
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
model.load_state_dict(torch.load(f'models/{model_path}.pth'))
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

# Check if a GPU is available and move the model to GPU if it is
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

embeddings = []
indices = []
countries = []

def process_data(data_partition):
    for row in tqdm(data_partition):
        fp = row["File Path"]
        
        # Open image and process
        image = Image.open(fp).convert("RGB")  # Ensure image is in RGB
        image_tensor = processor(images=image, return_tensors="pt").to(device)
        
        # Compute embedding
        with torch.no_grad():
            embedding = model.get_image_features(**image_tensor)
            
        # Move embedding back to CPU for numpy conversion and storage
        embedding = embedding.cpu().numpy()
        embeddings.append(embedding)
        indices.append(row["__index_level_0__"])

# Assuming load_google_data is defined elsewhere and loads your data
data = load_google_data("imbag_clip_dataset.hf")

# Process validation and train data
process_data(data["validation"])

# Stack embeddings and prepare for saving
embeddings_array = np.vstack(embeddings)
ids = np.array(indices, dtype=np.int64)

# Create a DataFrame with embeddings and their IDs
column_names = [f'dim{i}' for i in range(embeddings_array.shape[1])] + ['id']
df = pd.DataFrame(data=np.column_stack((embeddings_array, ids)), columns=column_names)

# Ensure the 'id' column is of integer type
df['id'] = df['id'].astype(np.int64)  # or np.int32 as needed

# Save to Parquet
df.to_parquet(f'/home/data_shares/geocv/{model_path}_embeddings_with_ids_val.parquet') 


embeddings = []
indices = []
countries = []

process_data(data["train"])

# Stack embeddings and prepare for saving
embeddings_array = np.vstack(embeddings)
ids = np.array(indices, dtype=np.int64)

# Create a DataFrame with embeddings and their IDs
column_names = [f'dim{i}' for i in range(embeddings_array.shape[1])] + ['id']
df = pd.DataFrame(data=np.column_stack((embeddings_array, ids)), columns=column_names)

# Ensure the 'id' column is of integer type
df['id'] = df['id'].astype(np.int64)  # or np.int32 as needed

# Save to Parquet
df.to_parquet(f'/home/data_shares/geocv/{model_path}_embeddings_with_ids_train.parquet') 