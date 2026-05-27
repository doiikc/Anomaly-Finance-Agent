import yfinance as yf
import pandas as pd
import numpy as np

def data_fetcher(stockName, period="1y", interval="1d"):
    data= yf.download(tickers= stockName, period=period, interval=interval)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    data=data[['Close']]
    return data

def detect_anomalies(data):
    data['Mean']= data['Close'].rolling(window=30).mean()
    data['Std']= data['Close'].rolling(window=30).std()

    data['Z_score']= (data['Close']-data['Mean'])/ data['Std']

    data['Is_Anomaly']= data['Z_score'].abs() >3
    return data
