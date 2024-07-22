import os
import json
import csv


def collect_all_keys(folder_path):
    all_keys = set()

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            
            # Open and read each JSON file
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    all_keys.update(data.keys())
                    # Collect nested keys
                    if 'thread' in data:
                        for key in data['thread'].keys():
                            all_keys.add(f"thread.{key}")
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {filename}")
                except Exception as e:
                    print(f"An error occurred with file {filename}: {e}")

    return sorted(all_keys)

def extract_data_from_folder(folder_path, all_keys):
    extracted_data = []

    # Iterate through all files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            
            # Open and read each JSON file
            with open(file_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    row = {}
                    for key in all_keys:
                        if '.' in key:
                            keys = key.split('.')
                            value = data.get(keys[0], {}).get(keys[1], "No Data")
                        else:
                            value = data.get(key, "No Data")
                        row[key] = value
                    extracted_data.append(row)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {filename}")
                except Exception as e:
                    print(f"An error occurred with file {filename}: {e}")

    return extracted_data

def save_to_csv(data, all_keys, csv_path):
    # Write data to CSV file
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.DictWriter(csvfile, fieldnames=all_keys)
        csvwriter.writeheader()
        csvwriter.writerows(data)

# Specify the path to your folder containing JSON files
folder_path = r'C:\Users\Asus\Documents\GitHub\Practiques_UDG\Stock\News articles\2015-07'
# Specify the path to save the CSV file
csv_path = r'C:\Users\Asus\Documents\GitHub\Practiques_UDG\Stock\News articles\all_data.csv'

# Collect all possible keys from the JSON files
all_keys = collect_all_keys(folder_path)

# Extract data from the JSON files using the collected keys
extracted_data = extract_data_from_folder(folder_path, all_keys)

# Save extracted data to CSV
save_to_csv(extracted_data, all_keys, csv_path)



