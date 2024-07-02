import tweepy

# Defineix les teves credencials directament al fitxer
consumer_key = '3Z8pCfLhnmZApN1c6b9ERI4RL'
consumer_secret = 'ke9xcfZaP4NquCfrJnZlFs9OQSJ442B6GVzAVZGkBsncLoUx5J'
access_token = '1434804700302528512-PXVQJRA8pRO2KRRdQA5v5HJaGBcgYd'
access_token_secret = '8LNv286hzQXW13zInCgMOmptePTxbBiVmPc9D3x5WMwqt'

# Autenticació amb la API de Twitter utilitzant Tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Crear un objecte API amb la versió 1.1 de la API de Twitter
api = tweepy.API(auth, wait_on_rate_limit=True)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print(f'{tweet.user.screen_name}:\n{tweet.text}\n{"*"*60}')

id = None
count = 0
while count <= 30:
    tweets = api.search(q='crypto', lang='en', tweet_mode='extended', max_id=id)
    for tweet in tweets:
        if tweet.full_text.startswith('RT'):
            count += 1
            continue
        with open('./onepiece.txt', 'a', encoding='utf-8') as f:
            f.write(tweet.full_text + '\n')
        count += 1
    id = tweet.id
    print(count)
