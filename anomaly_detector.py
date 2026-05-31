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
    data['Status_Reason']="Normal Market Activity"
    data['Is_Anomaly']=False

    price_error= data['Close'] <=0
    data.loc[price_error, 'Is_Anomaly']=True
    data.loc[price_error, 'Status_Reason']= "Bad Data:Price cannot be zero or negative"

    data['Mean']= data['Close'].rolling(window=30).mean()
    data['Std']= data['Close'].rolling(window=30).std()
    data['Z_score']= (data['Close']-data['Mean'])/ data['Std']

    z_error = (data['Z_score'].abs() > 3) & (data['Status_Reason'] == "Normal Market Activity")
    #normal market activity so that it isnt a logical error
    data.loc[z_error, 'Is_Anomaly'] = True
    data.loc[z_error, 'Status_Reason'] = "Statistical Anomaly: Significant price movement"
    return data

