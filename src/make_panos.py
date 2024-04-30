import cv2
import numpy as np
import os
from tqdm import tqdm

# Load the images
files = os.listdir('/home/data_shares/geocv/concat_google_images')

pano_path = "/home/data_shares/geocv/panos"

# get indices from file names
indices = []
for file in files:
    if file.endswith('.jpg'):
        indices.append(int(file.split('_')[1]))

# remove duplicates
indices = list(set(indices))

for index in tqdm(indices):
    try:
        # Create an empty list to store the images
        pano_images = []
        for i in range(0, 360, 90):
            img = cv2.imread(f'/home/data_shares/geocv/concat_google_images/google_{index}_{i}.jpg')
            pano_images.append(img)

        # Stitch images horizontally
        pano_stitch = np.concatenate(pano_images, axis=1)

        # Save the stitched image
        file_name_pano = f'{pano_path}/{index}.jpg'
        cv2.imwrite(file_name_pano, pano_stitch)
    except:
        print(f"Error in stitching {index}")
        continue