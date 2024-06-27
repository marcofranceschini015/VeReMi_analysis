import os
import pandas as pd
from tqdm import tqdm


"""
Reads all JSON files in a directory (and its subdirectories) that start with a given prefix.

Args:
folder_path (str): The path to the directory to search within.
file_prefix (str): The prefix of the files to read.

Returns:
pandas.DataFrame: A single DataFrame containing all data from the matching JSON files.
"""
def read_json_files(folder_path, file_prefix, read_vehicle_number):
    data_frames = []
    veh_number = 1
    # Walk through all directories and files in the folder_path
    for root, dirs, files in os.walk(folder_path):
        for file in tqdm(files, desc=f"Processing data - {file_prefix}"):
            # Check if the file starts with the desired prefix and ends with '.json'
            if file.startswith(file_prefix) and file.endswith(".json"):
                file_path = os.path.join(root, file)
                # Read the JSON file and append to the list of DataFrames
                df = pd.read_json(file_path, lines=True)
                if read_vehicle_number:
                    df['veh'] = veh_number
                    veh_number = veh_number + 1
                data_frames.append(df)
    print(f"\nFile read = {len(data_frames)}")
    return concatenate(data_frames)
    

def concatenate(data_frames):
    # Concatenate all DataFrames into one, if any exist
    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no files were found
