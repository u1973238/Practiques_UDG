import os
import csv

# Define the directory containing the CSV files
input_directory = 'Bitcoin'  # Replace with your folder path
output_file = 'Bitcoin_sentiment_summary.csv'

# Initialize a list to store the summary data
summary_data = []

# Iterate over each file in the directory
for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        file_path = os.path.join(input_directory, filename)
        
        # Extract the date from the filename (assuming format 'YYYY-MM-DD.csv')
        date = filename.split('.csv')[0]
        
        # Initialize counters for sentiment scores
        count_0 = 0
        count_1 = 0
        count_2 = 0
        
        # Open the CSV file and read its contents
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                sentiment = row['Sentiment Score']
                if sentiment == '0.0':
                    count_0 += 1
                elif sentiment == '1.0':
                    count_1 += 1
                elif sentiment == '2.0':
                    count_2 += 1
        
        # Add the results to the summary data
        summary_data.append([date, count_0, count_1, count_2])

# Define the header for the summary CSV
headers = ['Date', 'Count of 0', 'Count of 1', 'Count of 2']

# Write the summary data to the output CSV file
with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)
    writer.writerows(summary_data)

print(f'Summary has been written to {output_file}')
