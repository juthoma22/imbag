from ClipFineTuner import CLIPFineTuner
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
import wandb
from load_data import load_google_data
from PIL import Image

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


class ImbagClip():
    def __init__(self, mode="country_first"):
        """
        Take dataset configuration
        Load dataset in that configuration
        Initialize dataloader
        Initialize a CLIPFineTuner
        Train CLIPFineTuner on the dataset

        Args:
        mode (str): "country_first" for Country -> Climate -> Geocell, "climate_first" for Climate -> Country -> Geocell, "separate" for mixed in separate captions, "combined" for a single combined caption
        """
        self.mode = mode
        self.model = CLIPFineTuner() # TODO: add parameters
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336") 

        wandb.init() # add project name


    def load_dataset(self):
        """
        Load dataset from disk

        Depending on mode, load dataset in that configuration

        Returns:
        dataset: dataset object
        """
        dataset = load_google_data("Imbag")
        self.train_dataset, self.validation_dataset = dataset['train'], dataset['validation']


    def process_data(self, examples, argument):
        """
        Process the data for CLIP model

        Args:
        batch: batch of data

        Returns:
        batch: processed batch
        """
        captions = []
        images = []

        if argument == "Combined":
            for country, climate, geocell, file_path in zip(examples["Country"], examples["Climate Zone"], examples["Geocell"], examples["file_path"]):
                # Load image using PIL
                img = Image.open(file_path)
                images.append(img)

                caption = "this image was taken in " + country + ", specifically in Geocell " + geocell + " and in " + climate_zone_descriptions.get(climate).lower()
                captions.append(caption)

        else:
            for arg, file_path in zip(examples[argument], examples["file_path"]):
                # Load image using PIL
                img = Image.open(file_path)
                images.append(img)

                if argument == "Climate Zone":
                    caption = "this image was taken in " + climate_zone_descriptions.get(arg).lower()
                elif argument == "Country":
                    caption = "this image was taken in " + arg
                elif argument == "Geocell":
                    caption = "this image was taken in Geocell " + arg
                else:
                    print("Invalid argument")
                    continue

                captions.append(caption)

        return self.processor(text=captions, images=images, return_tensors="pt", padding="max_length", max_length=32, truncation=True)

        
    def preprocess(self):
        """
        Using CLIP Processor to preprocess the dataset

        Returns:
        dataset: dataset object
        """
        if self.mode == "combined":
            self.train_dataset = self.process_data(self.train_dataset, "Combined")
            self.validation_dataset = self.process_data(self.validation_dataset, "Combined")

        else:
            train_country_dataset = self.train_dataset.copy()
            train_climate_dataset = self.train_dataset.copy()
            train_geocell_dataset = self.train_dataset.copy()

            validation_country_dataset = self.validation_dataset.copy()
            validation_climate_dataset = self.validation_dataset.copy()
            validation_geocell_dataset = self.validation_dataset.copy()

            train_country_dataset = self.process_data(self.train_dataset, "Country")
            validation_country_dataset = self.process_data(self.validation_dataset, "Country")

            train_climate_dataset = self.process_data(self.train_dataset, "Climate Zone")
            validation_climate_dataset = self.process_data(self.validation_dataset, "Climate Zone")

            train_geocell_dataset = self.process_data(self.train_dataset, "Geocell")
            validation_geocell_dataset = self.process_data(self.validation_dataset, "Geocell")

            if self.mode == "country_first":
                self.train_dataset = train_country_dataset
                self.validation_dataset = validation_country_dataset
                self.train_dataset += train_climate_dataset
                self.validation_dataset += validation_climate_dataset
                self.train_dataset += train_geocell_dataset
                self.validation_dataset += validation_geocell_dataset

            elif self.mode == "climate_first":
                self.train_dataset = train_climate_dataset
                self.validation_dataset = validation_climate_dataset
                self.train_dataset += train_country_dataset
                self.validation_dataset += validation_country_dataset
                self.train_dataset += train_geocell_dataset
                self.validation_dataset += validation_geocell_dataset
            
            elif self.mode == "separate":
                self.train_dataset = train_country_dataset
                self.validation_dataset = validation_country_dataset
                self.train_dataset += train_climate_dataset
                self.validation_dataset += validation_climate_dataset
                self.train_dataset += train_geocell_dataset
                self.validation_dataset += validation_geocell_dataset

        self.train_dataset = self.processor(self.train_dataset, return_tensors="pt", padding=True)
        self.validation_dataset = self.processor(self.validation_dataset, return_tensors="pt", padding=True)


    def prepare_dataloader(self, batch_size=8):
        """
        Prepare dataloader for training
        """
        if self.mode == "separate":
            shuffle = True
        else:
            shuffle = False

        self.train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'pixel_values'])
        self.validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'pixel_values'])
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
        self.train_dataset = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


    def train(self):
        """
        Train the model on the dataset

        Returns:
        model: trained model
        """
        self.model.train(self.train_loader, self.validation_loader)
        self.model.save_model()


if __name__ == "__main__":
    imbag_clip = ImbagClip()
    imbag_clip.load_dataset()
    imbag_clip.preprocess()
    imbag_clip.train()