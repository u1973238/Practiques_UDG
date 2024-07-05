import time
from pytrends.request import TrendReq
import pandas as pd

# Configura la connexió amb Google Trends
pytrends = TrendReq(hl='ca', tz=360)

# Defineix els termes de cerca
keyword = "cryptocurrency"

# Crea una llista d'intervalls de temps des del 2020 fins al 2024
timeframes = [
    '2020-01-01 2020-12-31',
    '2021-01-01 2021-12-31',
    '2022-01-01 2022-12-31',
    '2023-01-01 2023-12-31',
    '2024-01-01 2024-07-01'  # Assumint que l'avui és el 1 de Juliol de 2024
]

# Llista per emmagatzemar els resultats
frames = []

# Funció per realitzar cerques amb retries
def get_interest_over_time(pytrends, timeframe, retries=5, backoff_factor=1):
    for i in range(retries):
        try:
            pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='', gprop='')
            return pytrends.interest_over_time()
        except pytrends.exceptions.TooManyRequestsError:
            if i < retries - 1:
                sleep_time = backoff_factor * (2 ** i)
                print(f"Too many requests. Sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                raise

# Realitza la cerca per a cada interval de temps
for timeframe in timeframes:
    try:
        interest_over_time_df = get_interest_over_time(pytrends, timeframe)
        frames.append(interest_over_time_df)
        # Dorm uns segons per evitar ser bloquejat per Google
        print("Sleeping for 60 seconds between requests...")
        time.sleep(60)
    except Exception as e:
        print(f"Failed to get data for timeframe {timeframe}: {e}")
        continue

# Combina tots els resultats en un únic DataFrame
combined_df = pd.concat(frames)

# Configura pandas per evitar el FutureWarning
pd.set_option('future.no_silent_downcasting', True)

# Mostra els resultats
print(combined_df)

# Guarda els resultats en un fitxer CSV
combined_df.to_csv('search_interest_daily_2020_2024.csv')
