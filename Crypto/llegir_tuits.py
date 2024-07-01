import tweepy

# Defineix les teves credencials directament al fitxer
consumer_key = '3Z8pCfLhnmZApN1c6b9ERI4RL'
consumer_secret = 'ke9xcfZaP4NquCfrJnZlFs9OQSJ442B6GVzAVZGkBsncLoUx5J'
access_token = '1434804700302528512-PXVQJRA8pRO2KRRdQA5v5HJaGBcgYd'
access_token_secret = '8LNv286hzQXW13zInCgMOmptePTxbBiVmPc9D3x5WMwqt'

# Autenticació amb la API de Twitter utilitzant Tweepy
client = tweepy.Client(bearer_token=bearer_token)

# Buscar tuits amb l'API v2 de Twitter
query = 'crypto -is:retweet lang:en'
tweets = client.search_recent_tweets(query=query, tweet_fields=['author_id', 'created_at'], max_results=100)

# Processar i mostrar els tuits
for tweet in tweets.data:
    print(f'Author ID: {tweet.author_id}\nCreated At: {tweet.created_at}\nTweet: {tweet.text}\n{"*"*60}')

# Emmagatzemar els tuits en un fitxer
count = 0
with open('./onepiece.txt', 'a', encoding='utf-8') as f:
    for tweet in tweets.data:
        f.write(tweet.text + '\n')
        count += 1
    print(count)