import tweepy
import json
import csv

# Credencials de l'API de Twitter
consumer_key = '3Z8pCfLhnmZApN1c6b9ERI4RL'
consumer_secret = 'ke9xcfZaP4NquCfrJnZlFs9OQSJ442B6GVzAVZGkBsncLoUx5J'
access_token = '1434804700302528512-PXVQJRA8pRO2KRRdQA5v5HJaGBcgYd'
access_token_secret = '8LNv286hzQXW13zInCgMOmptePTxbBiVmPc9D3x5WMwqt'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAGURugEAAAAANqP5%2Bs8UmwrbOsUgZWuwtOuj%2Fcs%3DFoMsmyHhmHv5M3dB4nzeb23LniAJRiLnfL8EcqlS2y3QrtoII9'

# Inicialitzar el fitxer CSV amb l'encapçalat si és la primera vegada
with open('tweets.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['created_at', 'user', 'text'])

# Definició del Listener per la Streaming API
class MyStreamListener(tweepy.StreamingClient):

    def on_data(self, data):
        try:
            tweet = json.loads(data)
            tweet_text = tweet['data']['text']
            tweet_created_at = tweet['data']['created_at']
            tweet_user = tweet['data']['author_id']
            
            # Escriure les dades del tuit en un fitxer CSV
            with open('tweets.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([tweet_created_at, tweet_user, tweet_text])
            
            print(f"Tweet by {tweet_user} at {tweet_created_at}: {tweet_text}")
            return True
        except BaseException as e:
            print(f'Error on_data: {str(e)}')
        return True

    def on_error(self, status):
        print(status)
        return True

# Crear un objecte API
client = tweepy.Client(bearer_token=bearer_token)

# Iniciar el Stream
myStreamListener = MyStreamListener(bearer_token)
myStreamListener.add_rules(tweepy.StreamRule("bitcoin OR ethereum OR cryptocurrency"))
myStreamListener.filter()

