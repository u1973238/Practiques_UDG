import requests
import csv
from googletrans import Translator
import stanza
from datetime import datetime, timedelta

# Set up the Stanza pipeline
stanza.download('en')  # Download the English models (do this only once)
nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')

translator = Translator()

# Define the API key and the base URL
API_KEY = 'b8caede577fe4225bd795bc19afbd226'
base_url = 'https://newsapi.org/v2/everything'

# Define the start and end dates for the range
start_date = datetime(2024, 7, 1)
end_date = datetime(2024, 7, 29)
date_generated = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') for x in range((end_date - start_date).days + 1)]

for date in date_generated:
    # Define the query parameters
    params = {
        'q': 'bitcoin',
        'from': date,
        'to': date,
        'sortBy': 'popularity',
        'apiKey': API_KEY
    }
    
    # Send GET request to the News API
    response = requests.get(base_url, params=params)
    data = response.json()
    articles = data.get('articles', [])

    # Define the file path for the CSV
    file_path = f'{date}.csv'

    # Define the CSV column headers
    headers = ['Title', 'Sentiment Score']

    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write the headers

        for article in articles:
            title = article['title']
            if isinstance(title, str):
                detected_language = translator.detect(title).lang
                if detected_language == 'en':
                    doc = nlp(title)
                    sentiment_score = sum([sentence.sentiment for sentence in doc.sentences]) / len(doc.sentences)
                    writer.writerow([title, sentiment_score])
