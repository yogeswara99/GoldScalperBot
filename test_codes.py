import yfinance as yf
import pandas as pd
from datetime import datetime

# SBI stock ticker on Yahoo Finance
ticker = "SBIN.NS"

# Date range
start_date = "2020-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

# Download data
data = yf.download(ticker, start=start_date, end=end_date)

# Upgrade: move Date from index to column
data.reset_index(inplace=True)

# File path
file_path = r"C:\Users\user\Downloads\SBI_Stock_Data.xlsx"

# Save to Excel
data.to_excel(file_path, index=False)

print(f"Data successfully downloaded to {file_path}")
