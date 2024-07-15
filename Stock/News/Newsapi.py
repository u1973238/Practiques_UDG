import requests

url = ('https://newsapi.org/v2/top-headlines?q=tesla&from=2024-06-10&sortBy=publishedAt&apiKey=b8caede577fe4225bd795bc19afbd226')

'''
keywords = ['Apple', 'iPhone', 'MacBook', 'AAPL', 'Samsung', 'Apple store', 
            'Apple Watch', 'Apple accessories', 'Tim Cook', 'iPad', 
            'Apple services', 'Apple support', 'Apple reviews']
'''

# Send GET request to the News API
response = requests.get(url)
data = response.json()
#print(data)

articles = data['articles']

file_path = 'NewsApi.txt'

with open(file_path, 'a', encoding='utf-8') as file:
    for article in articles:
                file.write(f"Title: {article['title']}\n")
                file.write(f"Published At: {article['publishedAt']}\n")
                file.write("\n")

