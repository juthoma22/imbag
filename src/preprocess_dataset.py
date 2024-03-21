
import os
import datasets
from datasets import Dataset, DatasetDict
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import sys

argument = sys.argv[1]
if argument == "climate_zone":
    argument = "Climate Zone"

processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

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

def load_google_data():
    path = "/home/data_shares/geocv/concat_dataset.hf"
    if os.path.exists(path):
        hf_dataset = datasets.load_from_disk(path)
        return hf_dataset
    else:
        print("No dataset found")

def load_and_preprocess_data():
    dataset = load_google_data()
    train_dataset, validation_dataset = dataset['train'], dataset['validation']
    workers = torch.cuda.device_count()
    preprocessed_train_dataset = train_dataset.map(process_data, batched=True, batch_size=500, num_proc=workers, remove_columns=["file_path", "Climate Zone", "Country"])
    preprocessed_validation_dataset = validation_dataset.map(process_data, batched=True, batch_size=500, num_proc=workers, remove_columns=["file_path", "Climate Zone", "Country"])
    return preprocessed_train_dataset, preprocessed_validation_dataset

def process_data(examples):
    captions = []
    images = []
    for arg, file_path in zip(examples[argument], examples["file_path"]):
        # Load image using PIL
        img = Image.open(file_path)
        images.append(img)

        if argument == "Climate Zone":
            caption = "this image was taken in " + climate_zone_descriptions.get(arg).lower()
        elif argument == "Country":
            caption = "this image was taken in " + arg
        else:
            print("Invalid argument")
            continue

        captions.append(caption)

    return processor(text=captions, images=images, return_tensors="pt", padding="max_length", max_length=32, truncation=True)


if __name__ == "__main__":
    preprocessed_train_dataset, preprocessed_validation_dataset = load_and_preprocess_data()
    preprocessed_full_dataset = DatasetDict({"train": preprocessed_train_dataset, "validation": preprocessed_validation_dataset})

    preprocessed_full_dataset.save_to_disk(f'/home/data_shares/geocv/preprocessed_{argument}_dataset.hf')