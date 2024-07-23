

import requests

# Make the GET request
url = ('https://api.worldnewsapi.com/top-news?api-key=a4efeedf68784748b1e17ed52dfcd954&source-country=us&language=en&topic=economic&date=2024-06-15')
#https://newsapi.org/v2/everything?q=apple&from=2024-07-21&to=2024-07-21&sortBy=popularity&apiKey=b8caede577fe4225bd795bc19afbd226

response = requests.get(url)
data = response.json()

# Extract the nested articles list
articles = data['top_news'][2]['news']

# Define the file path
file_path = 'WorldnewApi.txt'

# Write the data to the file
with open(file_path, 'w', encoding='utf-8') as file:
    for article in articles:
        # Check if 'title' and 'publish_date' keys exist in the article
        if 'title' in article and 'publish_date' in article:
            file.write(f"Title: {article['title']}\n")
            file.write(f"Published At: {article['publish_date']}\n")
            file.write(f"sentiment: {article['sentiment']}\n")
            file.write("\n")
        else:
            print(f"Missing 'title' or 'publish_date' in article: {article}")
