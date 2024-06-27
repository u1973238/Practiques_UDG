import tweepy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import numpy as np

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

# Descarregar recursos NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Funció per preprocessar els tweets
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'http\S+', '', text)  # Elimina URLs
    text = re.sub(r'[^\w\s]', '', text)  # Elimina caràcters especials
    tokens = word_tokenize(text.lower())  # Tokenització i conversió a minúscules
    tokens = [word for word in tokens if word.isalpha()]  # Només paraules alfabètiques
    tokens = [word for word in tokens if word not in stop_words]  # Elimina stopwords
    return tokens

# Definim la cadena de cerca
search_query = 'bitcoin OR ethereum OR cryptocurrency'

# Recopilem tweets relacionats amb criptomonedes amb l'API v1.1 de Tweepy
tweets = api.search(q=search_query, tweet_mode='extended', count=1000)

tweet_texts = []
for tweet in tweets:
    tweet_texts.append(tweet.full_text)


# Preprocessa els tweets
cleaned_tweets = [preprocess_text(tweet) for tweet in tweet_texts]

# Crea un diccionari de paraules cripto (per simplificar, aquí es fa manualment)
crypto_dict = {
    'bitcoin': 0,
    'ethereum': 1,
    'cryptocurrency': 2
}

# Crea vectors de característiques per a cada tweet
def create_features(tweet_tokens, crypto_dict):
    features = np.zeros((len(tweet_tokens), len(crypto_dict)))
    for i, tokens in enumerate(tweet_tokens):
        for token in tokens:
            if token in crypto_dict:
                features[i, crypto_dict[token]] += 1
    return features

# Crea vectors de característiques
features = create_features(cleaned_tweets, crypto_dict)

# Conversió a numpy array per al model
X = np.array(features)

# Labels de sentiments (dummy per a aquest exemple)
y = np.random.randint(3, size=len(features))  # Dummy labels, has d'assignar els sentiments correctament

# Divisió de les dades en conjunts d'entrenament i prova
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construcció i configuració del model de xarxa neuronal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

embedding_dim = 100
max_length = X.shape[1]

model = Sequential([
    Embedding(len(crypto_dict), embedding_dim, input_length=max_length),
    SpatialDropout1D(0.2),
    LSTM(100, dropout=0.2, recurrent_dropout=0.2),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenament del model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Avaluació del model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

# Funció per predir el sentiment de nous tweets
def predict_sentiment(tweet, model, tokenizer, max_length):
    cleaned_tweet = preprocess_text(tweet)
    tokenized = tokenizer.texts_to_sequences([cleaned_tweet])
    padded = pad_sequences(tokenized, maxlen=max_length)
    prediction = model.predict(padded)
    sentiment = np.argmax(prediction)
    return sentiment

# Exemple de com utilitzar la funció de predicció
new_tweet = "Bitcoin is making great progress in the market today."
predicted_sentiment = predict_sentiment(new_tweet, model, None, max_length)
sentiment_labels = ['Negative', 'Neutral', 'Positive']
print(f'Predicted sentiment: {sentiment_labels[predicted_sentiment]}')

