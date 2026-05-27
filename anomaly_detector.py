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

if __name__ == "__main__":
    # 1. Veriyi çekiyoruz
    ham_veri = data_fetcher("TSLA")
    
    # 2. Anomalileri tespit ediyoruz
    analiz_edilmis_veri = detect_anomalies(ham_veri)
    
    # 3. Sadece anomali olan (True) satırları filtreleyip ekrana yazdıralım
    anomaliler = analiz_edilmis_veri[analiz_edilmis_veri['Is_Anomaly'] == True]
    
    print("--- TESPİT EDİLEN ANOMALİLER ---")
    print(anomaliler)