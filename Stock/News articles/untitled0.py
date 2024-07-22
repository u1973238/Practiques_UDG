import os
import json
import csv

def read_json_files(folder_path):
    data = []
    # List all files in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                json_data = json.load(file)
                data.append(json_data)
    return data

def write_to_csv(data, output_file):
    if not data:
        return
    
    # Extract fieldnames from the first data entry
    fieldnames = data[0].keys()
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in data:
            writer.writerow(entry)

def main():
    folder_path = r'C:\Users\Asus\Documents\GitHub\Practiques_UDG\Stock\News articles\2015-07'  # Replace with the path to your folder containing JSON files
    output_file = 'output.csv'  # Replace with the desired output CSV file name
    
    data = read_json_files(folder_path)
    write_to_csv(data, output_file)
    print(f'Data successfully written to {output_file}')

if __name__ == "__main__":
    main()
