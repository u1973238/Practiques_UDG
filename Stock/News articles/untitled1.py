import csv

def read_csv(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            data.append(row)
    return data

def main():
    file_path = r'C:\Users\Asus\Documents\GitHub\Practiques_UDG\Stock\News articles\output.csv'  # Replace with the path to your CSV file
    csv_data = read_csv(file_path)
    print(csv_data)

if __name__ == "__main__":
    main()
