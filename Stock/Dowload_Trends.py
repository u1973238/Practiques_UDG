from pytrends.request import TrendReq
import time
import pandas as pd
import yfinance as yf

# Downloading Daily Historical Stock Data
stock_symbol = 'AAPL'
start_date = '2023-07-01'
end_date = '2024-07-01'

# Function to fetch Google Trends data with retry mechanism
def fetch_google_trends_data(keyword, start_date, end_date):
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([keyword], timeframe=f'{start_date} {end_date}')
    trends_data = None
    while trends_data is None or trends_data.empty:
        try:
            trends_data = pytrends.interest_over_time()
            if trends_data.empty:
                print("Google Trends data is empty, retrying in 20 seconds...")
                time.sleep(20)
        except Exception as e:
            print(f"Error: {e}, retrying in 20 seconds...")
            time.sleep(20)
    return trends_data[[keyword]]

# List of keywords to fetch Google Trends data for
keywords = ['Apple', 'iPhone', 'MacBook', 'AAPL', 'Samsung', 'Apple store', 
            'Apple Watch', 'Apple accessories', 'Tim Cook', 'iPad', 
            'Apple services', 'Apple support', 'Apple reviews']

# Fetching and normalizing Google Trends data
trends_data_list = []
for keyword in keywords:
    trends_data = fetch_google_trends_data(keyword, start_date, end_date)
    pd_trends_data = pd.DataFrame(trends_data)
    pd_trends_data = pd_trends_data.resample('B').interpolate()

    # Min-Max normalization of the trend data
    min_val = pd_trends_data[keyword].min()
    max_val = pd_trends_data[keyword].max()
    pd_trends_data[keyword] = (pd_trends_data[keyword] - min_val) / (max_val - min_val)
    
    trends_data_list.append(pd_trends_data)

# Combining the Google Trends data
combined_trends_data = pd.concat(trends_data_list, axis=1)

stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Creating a date range for the entire period
full_date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # 'B' frequency is for business days

# Reindexing the stock data to include all dates in the full date range and using forward fill
stock_data = stock_data.reindex(full_date_range).ffill()

# Merging the stock data with the combined Google Trends data
merged_data = stock_data.join(combined_trends_data, how='inner')

# Creating the target column
merged_data['target'] = merged_data['Close'].shift(-1) - merged_data['Close']
'''
# Min-Max normalization of the target column
min_target = merged_data['target'].min()
max_target = merged_data['target'].max()
merged_data['normalized_target'] = 2 * (merged_data['target'] - min_target) / (max_target - min_target) - 1
'''
# Saving the DataFrame to a CSV file
merged_data.to_csv('merged_data_with_normalized_target.csv', index=True)

