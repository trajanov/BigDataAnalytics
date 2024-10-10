# GDELT https://www.gdeltproject.org/ is a global event database that extracts events from news articles around the world
# The dataset is updated every 15 minutes and is available in CSV format
# The dataset contains information about the events such as the date, actors involved, location, and more
# The dataset is available at http://data.gdeltproject.org/gdeltv2/lastupdate.txt

import requests
import os
import re
import time
import zipfile
from urllib.parse import urljoin

# Define URL for GDELT last update file
lastupdate_url = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"

# Define the input path where the CSV file should be saved
input_path = "input_files"

# Function to check and download the new export.csv file
def check_and_download_file():
    try:
        # Fetch the last update information
        response = requests.get(lastupdate_url)
        response.raise_for_status()
        last_update_content = response.text.strip()
        
        # Use regex to find the URL of the export CSV file
        match = re.search(r'\bhttp://data\.gdeltproject\.org/gdeltv2/\d+\.export\.CSV\.zip\b', last_update_content)
        if match:
            csv_url = match.group(0)
            
            # Extract file name from URL
            file_name = csv_url.split("/")[-1]
            dest_path = os.path.join(input_path, file_name)
            
            # Check if the file already exists
            if not os.path.exists(dest_path):
                download_file(csv_url, dest_path)
                unzip_file(dest_path, input_path)
            else:
                print("File already exists at", dest_path)
    except requests.exceptions.RequestException as e:
        print("Error fetching last update info:", e)

# Function to download the file from the given URL
def download_file(url, dest_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Download the file and save it to the destination path
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print("Downloaded file to", dest_path)
    except requests.exceptions.RequestException as e:
        print("Error downloading the file:", e)

# Function to unzip the downloaded file
def unzip_file(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Unzipped file to", extract_to)
    except zipfile.BadZipFile as e:
        print("Error unzipping the file:", e)

if __name__ == "__main__":
    while True:
        check_and_download_file()
        time.sleep(300)  # Wait for 5 minutes before checking again