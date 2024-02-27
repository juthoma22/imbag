import time
import requests
import os
import cv2
import csv
import numpy as np
import argparse
import pandas as pd
import re

#api_key = os.environ.get('GOOGLEAPI')

def get_next_index(folder_path):
    """
    Get the next available index for saving Google images in a specified folder.

    Parameters:
    - folder_path (str): The path to the folder where Google images are stored.

    Returns:
    - int: The next available index.
    """
    pattern = re.compile(r'google_(\d+)_(\d+).jpg')
    files = [f for f in os.listdir(folder_path) if pattern.match(f)]
    indices = [int(pattern.match(f).group(1)) for f in files]
    max_index = max(indices, default=-1)
    return max_index + 1


def distribute_images(df, total_images, key='Country', factor=1, div=1):
    # Calculate the total length
    total_length = df['Length'].sum()

    # Calculate the total length for each country
    country_lengths = df.groupby(key)['Length'].sum()

    # Calculate the proportion of each country's length to the total length
    proportions = country_lengths / total_length

    # Calculate the number of images for each country
    images_per_country = (proportions * total_images).round().astype(int)

    # calculate minimum and maximum number of images per country
    min_images = max(images_per_country.min(),1)
    max_images = images_per_country.max()

    # Adjust the number of images for each country to have sufficient amounts for small countries
    images_per_country = images_per_country.clip(min_images*factor, max_images/div).astype(int)

    # Adjust the number of images for each country so that the total adds up to total_images
    diff = total_images - images_per_country.sum()
    while diff > 0:
        images_per_country += 1
        diff = total_images - images_per_country.sum()

    # Create a dictionary with country names as keys and the number of images as values
    images_dict = images_per_country.to_dict()

    return images_dict



def scrape_images(lat, lon, country, region, api_key, image_folder="/home/data_shares/geocv/more_google_images_1"):
    print(f"Scraping for {country} {region}, {lat}, {lon}...")

    os.makedirs(image_folder, exist_ok=True)
    start_index = get_next_index(image_folder)

    url = 'https://maps.googleapis.com/maps/api/streetview'
    num_rotations = 4
    rotation_angle = 90

    # make call to get metadata
    params = {'size': '640x640', 'location': f'{lat},{lon}', "radius":100, 'fov': '90', 'key': api_key}
    metadata_url = url + '/metadata?'
    metadata_response = requests.get(metadata_url, params=params)
    metadata = metadata_response.json()
    location = metadata['location']
    date = metadata['date']
    pano_id = metadata['pano_id']
    copyright = metadata['copyright']

    # take images angled down for every tenth location for car meta
    pitch = -30 if start_index % 10 == 0 else 0

    metadata = []
    for i in range(num_rotations):
        heading = str(i * rotation_angle)
        params = {'size': '640x640', 'location': f'{lat},{lon}', 'heading': heading, 'radius' : 100, 'fov': '90', 'pitch': pitch, 'key': api_key}

        try:
            response = requests.get(url, params=params)
        except Exception as e:
            time.sleep(5)
            try:
                response = requests.get(url, params=params)
            except Exception as e:
                print(e)
                print(f"Could not get image {lat} {lon} {country} {region}. Skipping...")
                continue
        try:
            image = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            file_name_singles = os.path.normpath(f'{image_folder}/google_{start_index}_{heading}.jpg')
            cv2.imwrite(file_name_singles, image)
        except cv2.error as e:
            print(f"Error saving image: {file_name_singles} ({e})")
            continue

        # Store metadata
        metadata.append([f"{start_index}_{heading}", location["lng"], location["lat"], pano_id, country, region, date, np.NaN])

    metadata_file_path = f"/home/data_shares/geocv/more_google_image_metadata_1.csv"

    file_exists = os.path.isfile(metadata_file_path)
    file_is_empty = not file_exists or os.path.getsize(metadata_file_path) == 0

    # Write metadata to a CSV file
    with open(metadata_file_path, mode="a+", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        # Write the header only if the file is empty
        if file_is_empty:
            writer.writerow(["Index", "Longitude", "Latitude", "Pano ID", "Country", "Region", "Date", "Climate Zone"])
        writer.writerows(metadata)

def meta_data_call(lat, lon, api_key):
    url = 'https://maps.googleapis.com/maps/api/streetview'
    params = {'size': '640x640', 'location': f'{lat},{lon}', "radius":100, 'fov': '90', 'key': api_key}

    # check for metadata
    metadata_url = url + '/metadata?'
    metadata_response = requests.get(metadata_url, params=params)
    metadata = metadata_response.json()
    if metadata['status'] != 'OK' or "© Google" not in metadata['copyright']:
        return -1
    
    return 0

def process_dataframe(df_geo, lengths, api_key, total_images=100000):

    images_per_country = distribute_images(lengths, total_images, key='Country', factor=10, div=2)
    # divide by 4 to get the number of images per region
    print(f"Total number of images: {sum(images_per_country.values())}")
    images_per_country = {k: v // 4 for k, v in images_per_country.items()}
    print(images_per_country)
    print(f"Total number of locations: {sum(images_per_country.values())}")

    # Add a new column 'processed_google' and set its default value to False
    if "processed_google" not in df_geo.columns:
        df_geo["processed_google"] = False

    # read csv file of processed regions and countries
    processed_regions = pd.read_csv("data/processed_regions_google_1.csv")
    if "Processed" not in processed_regions.columns:
        processed_regions["Processed"] = False
        processed_regions.to_csv("data/processed_regions_google_1.csv", index=False)

    for country, country_df in df_geo.groupby('Country'):
        print(f"Processing {country}...")
        lengths_per_country = lengths[lengths['Country'] == country]
        if country_df["Region"].unique()[0] == None:
            print("No regions found for this country. Processing country as a whole...")

            # check if region has already been processed in a previous run
            if processed_regions[processed_regions["Country"] == country]["Processed"].item():
                continue

            images_scraped = 0
            # Select a random point from the country_df for images_per_region[region] times
            while images_scraped < images_per_country[country]:
                # Skip the point if it has already been processed
                unprocessed_points = df_geo.loc[country_df.index][~df_geo.loc[country_df.index, 'processed_google']]
                if unprocessed_points.empty:
                    break

                attempts = 0
                max_attempts = 100 # keep track of attempts, if too many empty hits skip the region
                while attempts < max_attempts:
                    selected_point = unprocessed_points.sample(n=1)

                    lon, lat = selected_point['Longitude'].values[0], selected_point['Latitude'].values[0]

                    # Mark the matching points as processed
                    df_geo.loc[selected_point.index, 'processed_google'] = True

                    # add scraping call
                    val = meta_data_call(lat, lon, api_key)

                    if val == -1:
                        attempts += 1
                        continue

                    scrape_images(lat, lon, country, None, api_key)

                    attempts = 0  # Reset the attempts counter if a non-empty gdf is returned
                    images_scraped += 1 # adds the number of scraped images to the total

                    break  # Break the inner loop if a non-empty gdf is returned

                if attempts == max_attempts:
                    print("Reached maximum attempts for this country. Moving to the next country.")
                    break  # Break the outer loop if maximum attempts are reached
        
            # set country as processed
            processed_regions.loc[processed_regions["Country"] == country, "Processed"] = True
            processed_regions.to_csv("data/processed_regions_google_1.csv", index=False)
            # set points as processed
            df_geo.reset_index(drop=True).to_feather("data/coords_new.feather")
        else:
            images_per_region = distribute_images(lengths_per_country, images_per_country[country], key='Region', factor=1, div=1)
            print(images_per_region)
            
            # check if country has already been processed in a previous run
            all_processed = processed_regions.groupby('Country')['Processed'].all()
            if all_processed[country]:
                continue

            for region, region_df in country_df.groupby('Region'):
                print(f"Processing {region}...")

                # check if region has already been processed in a previous run
                if processed_regions[(processed_regions["Region"] == region) & (processed_regions["Country"] == country)]["Processed"].item():
                    continue

                images_scraped = 0
                # Select a random point from the region_df for images_per_region[region] times
                while images_scraped < images_per_region[region]:
                    # Skip the point if it has already been processed
                    unprocessed_points = df_geo.loc[region_df.index][~df_geo.loc[region_df.index, 'processed_google']]
                    if unprocessed_points.empty:
                        break

                    attempts = 0
                    max_attempts = 100 # keep track of attempts, if too many empty hits skip the region
                    while attempts < max_attempts:
                        selected_point = unprocessed_points.sample(n=1)

                        lon, lat = selected_point['Longitude'].values[0], selected_point['Latitude'].values[0]

                        # Mark the matching points as processed
                        df_geo.loc[selected_point.index, 'processed_google'] = True

                        # add scraping call
                        val = meta_data_call(lat, lon, api_key)

                        if val == -1:
                            attempts += 1
                            continue

                        scrape_images(lat, lon, country, region, api_key)

                        attempts = 0  # Reset the attempts counter if a non-empty gdf is returned
                        images_scraped += 1 # adds the number of scraped images to the total
                        break  # Break the inner loop if a non-empty gdf is returned

                    if attempts == max_attempts:
                        print("Reached maximum attempts for this region. Moving to the next region.")
                        break  # Break the outer loop if maximum attempts are reached
        
                # set region as processed
                processed_regions.loc[(processed_regions["Region"] == region) & (processed_regions["Country"]==country), "Processed"] = True
                processed_regions.to_csv("data/processed_regions_google_1.csv", index=False)
                # set points as processed
                df_geo.reset_index(drop=True).to_feather("data/coords_new.feather")
            # set country as processed
            processed_regions.loc[processed_regions["Country"] == country, "Processed"] = True
            processed_regions.to_csv("data/processed_regions_google_1.csv", index=False)


def main(file, n, api_key):
    df = pd.read_feather(file)
    lengths = pd.read_csv("data/road_lengths.csv")
    process_dataframe(df, lengths, api_key, n)
    print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scraping images from Google Street View')
    parser.add_argument('-f', '--file', type=str, required=True, help='File with coordinates')
    parser.add_argument('-n', '--number', type=int, required=True, help='Number of images to scrape')
    parser.add_argument('-a', '--api', type=str, required=True, help='api key')
    args = parser.parse_args()
    file = args.file
    n = args.number
    api_key = args.api
    main(file, n, api_key)