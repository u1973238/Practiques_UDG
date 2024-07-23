import requests
import csv
from googletrans import Translator
'''
url = ('https://newsapi.org/v2/everything?q=tesla&from=2024-07-20&to=2024-07-20&sortBy=publishedAt&apiKey=b8caede577fe4225bd795bc19afbd226')

# Send GET request to the News API
response = requests.get(url)
data = response.json()

articles = data['articles']
'''
file_path = '2024-07-20.csv'
'''
# Define the CSV column headers
headers = ['Title']

translator = Translator()

with open(file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write the headers

    for article in articles:
        title = article['title']
        if(isinstance(title, str)):
            d=translator.detect(title)
            if(d.lang=='en'):
                writer.writerow([title])  # Write each article as a row
'''
import pandas as pd
# Read the CSV file
csv_data = pd.read_csv(file_path)
