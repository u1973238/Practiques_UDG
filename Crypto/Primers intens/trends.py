import time
from pytrends.request import TrendReq
import pandas as pd
from pytrends.exceptions import TooManyRequestsError

# Configura la connexió amb Google Trends
pytrends = TrendReq(hl='ca', tz=360)

# Defineix els termes de cerca
keyword = "cryptocurrency"

# Data d'inici i finalització per a l'últim any
start_date = '2024-04-16'
end_date = '2024-07-16'

# Generem una llista de dates setmanals
dates = pd.date_range(start=start_date, end=end_date, freq='W-MON')

# Convertim les dates a intervals setmanals
timeframes = []
for i in range(len(dates) - 1):
    interval = f"{dates[i].strftime('%Y-%m-%d')} {dates[i + 1].strftime('%Y-%m-%d')}"
    timeframes.append(interval)

# Llista per emmagatzemar els resultats
frames = []

# Funció per realitzar cerques amb retries
def get_interest_over_time(pytrends, timeframe, retries=3, backoff_factor=2):
    for i in range(retries):
        try:
            pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='', gprop='')
            return pytrends.interest_over_time()
        except TooManyRequestsError as e:
            if i < retries - 1:
                sleep_time = backoff_factor * (2 ** i)
                print(f"Too many requests ({e}). Sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Failed to get data for timeframe {timeframe}: {e}")
                return None

# Realitza la cerca per a cada interval de temps
for timeframe in timeframes:
    interest_over_time_df = get_interest_over_time(pytrends, timeframe)
    if interest_over_time_df is not None:
        frames.append(interest_over_time_df)
    # Dorm uns segons per evitar ser bloquejat per Google
    print("Sleeping for 120 seconds between requests...")
    time.sleep(120)

# Combina tots els resultats en un únic DataFrame
if frames:
    combined_df = pd.concat(frames)
    # Mostra els resultats
    print(combined_df)
    # Guarda els resultats en un fitxer CSV
    combined_df.to_csv('trends.csv')
else:
    print("No data retrieved.")
