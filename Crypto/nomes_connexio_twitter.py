
##################################  FUNCIONA !!!! ################################## 

import tweepy

# Credencials de l'API de Twitter
consumer_key = '3Z8pCfLhnmZApN1c6b9ERI4RL'
consumer_secret = 'ke9xcfZaP4NquCfrJnZlFs9OQSJ442B6GVzAVZGkBsncLoUx5J'
access_token = '1434804700302528512-PXVQJRA8pRO2KRRdQA5v5HJaGBcgYd'
access_token_secret = '8LNv286hzQXW13zInCgMOmptePTxbBiVmPc9D3x5WMwqt'

# Autenticació amb l'API de Twitter
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Crear un objecte API
api = tweepy.API(auth)

# Prova de connexió: Obté la informació del teu perfil
try:
    user = api.verify_credentials()
    print(f'Connexió exitosa com a {user.screen_name}')

except Exception as e:
    print(f'Error en la connexió: {str(e)}')


