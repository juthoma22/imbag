import pandas as pd
from PIL import Image
import datasets
from datasets import Dataset, DatasetDict
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

df = pd.read_csv('/home/data_shares/geocv/google_image_metadata.csv')

df["file_name"] = df["Index"].apply(lambda x: f"google_{x}.jpg")

df = df[["file_name", "Climate Zone", "Country"]]

# df = df.dropna()

print(df.head())

# # make csv file from df
df.to_csv("/home/data_shares/geocv/google_images/singles/metadata.csv", index=False)

dataset = load_dataset("imagefolder", data_dir='/home/data_shares/geocv/google_images/singles/', split=['train', 'validation'])

split_datasets = DatasetDict({
    "train": dataset[0],
    "validation": dataset[1]
})

print(split_datasets)
print(split_datasets["train"])

split_datasets.save_to_disk('datasets/google_climate_dataset_split.hf')