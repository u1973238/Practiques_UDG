import requests

# Define the URL with parameters
url = ('https://newsapi.org/v2/top-headlines?'
       'q=Apple&'
       'from=2024-06-27&'
       'sortBy=popularity&'
       'apiKey=ebad4ad5bcbd4023816e0530da13504f')

url = ('https://newsapi.org/v2/top-headlines?q=stock&pageSize=100&apiKey=b8caede577fe4225bd795bc19afbd226')

#url = ('https://api.worldnewsapi.com/top-news?api-key=a4efeedf68784748b1e17ed52dfcd954&source-country=us&language=en&date')

#https://newsapi.org/v2/top-headlines?sources=bbc-news&apiKey=b8caede577fe4225bd795bc19afbd226
# Send GET request to the News API
response = requests.get(url)
data = response.json()
#print(data)

articles = data['articles']

file_path = 'NewsApi.txt'

with open(file_path, 'w', encoding='utf-8') as file:
    for article in articles:
                file.write(f"Title: {article['title']}\n")
                file.write(f"Published At: {article['publishedAt']}\n")
                file.write("\n")

# Check if the request was successful (status code 200)
#if response.status_code == 200:
    # Parse JSON response
#    data = response.json()
    # Check if there are articles in the response
#    if data['status'] == 'ok' and data['totalResults'] > 0:
#        articles = data['articles']

        # File path to write the output
#        file_path = 'top_headlines.txt'

        # Open the file in write mode
#        with open(file_path, 'w', encoding='utf-8') as file:
#            for article in articles:
#                file.write(f"Title: {article['title']}\n")
                #file.write(f"Source: {article['source']['name']}\n")
#                file.write(f"Published At: {article['publishedAt']}\n")
                #file.write(f"Description: {article['description']}\n")
#                file.write("\n")
        
#        print(f"Top headlines written to {file_path}")
#    else:
#        print("No articles found matching the criteria.")
#else:
#    print(f"Error fetching news: {response.status_code}")


#print(f"Top headlines written to {file_path}")
