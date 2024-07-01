import time
from pytrends.request import TrendReq
import pandas as pd

# Configura la connexió amb Google Trends
pytrends = TrendReq(hl='ca', tz=360)

# Defineix els termes de cerca
keyword = "cryptocurrency"

# Realitza una cerca diària per a la paraula clau en l'últim any
pytrends.build_payload([keyword], cat=0, timeframe='today 12-m', geo='', gprop='')

# Obté l'interès de cerca diari en l'últim any (12 mesos)
interest_over_time_df = pytrends.interest_over_time()

# Configura pandas per evitar el FutureWarning
pd.set_option('future.no_silent_downcasting', True)

# Mostra els resultats
print(interest_over_time_df)

# Guarda els resultats en un fitxer CSV
interest_over_time_df.to_csv('search_interest_daily.csv')
