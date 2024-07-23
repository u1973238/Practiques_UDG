
url = ('https://newsapi.org/v2/everything?q=tesla&from=2024-06-10&sortBy=publishedAt&apiKey=b8caede577fe4225bd795bc19afbd226')

import requests

api_key = 'b8caede577fe4225bd795bc19afbd226'
base_url = 'https://newsapi.org/v2/top-headlines?q=tesla&from=2024-06-10&sortBy=publishedAt&apiKey=b8caede577fe4225bd795bc19afbd226'

page = 1
all_articles = []

while True:
    url = f"{base_url}&page={page}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        articles = data["articles"]
        
        if not articles:  # No more articles left to fetch
            break
        
        all_articles.extend(articles)
        page += 1
    else:
        print(f"Error: {response.status_code}")
        break

print(f"Total number of articles retrieved: {len(all_articles)}")

file_path = 'NewsApi3.txt'

with open(file_path, 'a', encoding='utf-8') as file:
    for article in all_articles:
        file.write(f"Title: {article['title']}\n")
        file.write(f"Published At: {article['publishedAt']}\n")
        file.write("\n")
https://newsapi.org/v2/top-headlines?q=tesla&from=2024-06-10&sortBy=publishedAt&apiKey=b8caede577fe4225bd795bc19afbd226