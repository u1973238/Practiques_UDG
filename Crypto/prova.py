import tweepy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Credenciales de la API de Twitter
consumer_key = 'tu_consumer_key'
consumer_secret = 'tu_consumer_secret'
access_token = 'tu_access_token'
access_token_secret = 'tu_access_token_secret'

# Autenticación con la API de Twitter usando Tweepy
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Crear un objeto API con la versión 2 de la API de Twitter
api = tweepy.Client(bearer_token='tu_bearer_token', wait_on_rate_limit=True)

# Probar la conexión
try:
    user = api.get_user("username")
    print(f'Conexión exitosa como {user.screen_name}')
except Exception as e:
    print(f'Error en la conexión: {str(e)}')

# Descargar recursos NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Función para preprocesar los tweets
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'http\S+', '', text)  # Eliminar URLs
    text = re.sub(r'[^\w\s]', '', text)  # Eliminar caracteres especiales
    tokens = word_tokenize(text.lower())  # Tokenización y conversión a minúsculas
    tokens = [word for word in tokens if word.isalpha()]  # Solo palabras alfabéticas
    tokens = [word for word in tokens if word not in stop_words]  # Eliminar stopwords
    return tokens

# Definir la cadena de búsqueda
search_query = 'bitcoin OR ethereum OR cryptocurrency'

# Recopilar tweets relacionados con criptomonedas usando la nueva API v2
tweets = api.search_recent_tweets(query=search_query, tweet_mode='extended', max_results=1000)

tweet_texts = []
for tweet in tweets.data:
    tweet_texts.append(tweet.text)

# Preprocesar los tweets
cleaned_tweets = [preprocess_text(tweet) for tweet in tweet_texts]

# Crear un diccionario de palabras cripto
crypto_dict = {
    'bitcoin': 0,
    'ethereum': 1,
    'cryptocurrency': 2
}

# Crear características para cada tweet
def create_features(tweet_tokens, crypto_dict):
    features = np.zeros((len(tweet_tokens), len(crypto_dict)))
    for i, tokens in enumerate(tweet_tokens):
        for token in tokens:
            if token in crypto_dict:
                features[i, crypto_dict[token]] += 1
    return features

# Crear características
features = create_features(cleaned_tweets, crypto_dict)

# Conversión a array numpy para el modelo
X = np.array(features)

# Etiquetas de sentimientos (simulación aleatoria)
y = np.random.randint(3, size=len(features))  # Etiquetas aleatorias, deberás asignarlas correctamente

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configuración del modelo de red neuronal
embedding_dim = 100
max_length = X.shape[1]

model = Sequential([
    Embedding(len(crypto_dict), embedding_dim, input_length=max_length),
    SpatialDropout1D(0.2),
    LSTM(100, dropout=0.2, recurrent_dropout=0.2),
    Dense(3, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenamiento del modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluación del modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Pérdida: {loss}')
print(f'Precisión: {accuracy}')

# Función para predecir el sentimiento de nuevos tweets
def predict_sentiment(tweet, model, tokenizer, max_length):
    cleaned_tweet = preprocess_text(tweet)
    tokenized = tokenizer.texts_to_sequences([cleaned_tweet])
    padded = pad_sequences(tokenized, maxlen=max_length, padding='post')  # Asegúrate de usar pad_sequences aquí
    prediction = model.predict(padded)
    sentiment = np.argmax(prediction)
    return sentiment

# Ejemplo de cómo utilizar la función de predicción
new_tweet = "Bitcoin is making great progress in the market today."
predicted_sentiment = predict_sentiment(new_tweet, model, None, max_length)
sentiment_labels = ['Negativo', 'Neutral', 'Positivo']
print(f'Sentimiento predicho: {sentiment_labels[predicted_sentiment]}')
