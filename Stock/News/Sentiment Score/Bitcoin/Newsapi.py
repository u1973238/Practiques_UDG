
import requests
import csv
from googletrans import Translator
import stanza

url = ('https://newsapi.org/v2/everything?q=bitcoin&from=2024-08-21&to=2024-08-21&sortBy=popularity&apiKey=b8caede577fe4225bd795bc19afbd226')

# Send GET request to the News API
response = requests.get(url)
data = response.json()

articles = data['articles']

file_path = '2024-08-21.csv'

# Define the CSV column headers
headers = ['Title', 'Sentiment Score']

nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')

translator = Translator()

with open(file_path, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(headers)  # Write the headers

    for article in articles:
        title = article['title']
        if isinstance(title, str):
            #detected_language = translator.detect(title).lang
            #if detected_language == 'en':
            doc = nlp(title)
            sentiment_score = sum([sentence.sentiment for sentence in doc.sentences]) / len(doc.sentences)
            writer.writerow([title, sentiment_score])
