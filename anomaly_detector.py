import yfinance as yf
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from tavily import TavilyClient
from groq import Groq
import matplotlib.pyplot as plt

load_dotenv()
tavily= TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

def search_news(stock_name,date):
    query= f"{stock_name} stock market news catalyst event around {date}"
    print(f"Agent is searching internet for:{query}")

    try:
        response = tavily.search(query=query, max_results=3)
        context= ""

        for result in response.get('results', []):
            context += f"Title: {result['title']}\nSnippet: {result['content']}\nURL: {result['url']}\n---\n"

        return context if context else "No relevant news found for this date."
        
    except Exception as e:
        return f"Search failed due to error: {str(e)}"
    
def generate_analysis(stock_name, date, price, z_score, news_context):
    print("Agent is analyzing the news and creating a report...")


    system_prompt = """You are an experienced financial market analyst.

You will receive a list of events that occurred on a specific day, along with market movements

Your task is to:
1. Analyze each event and determine its likely impact on the observed market movements.
2. Identify the most influential events and explain the transmission mechanism from the event to the market reaction.
3. Distinguish between correlation and likely causation. Do not claim causality unless it is strongly supported by the information provided.
4. If multiple events could explain the same movement, discuss all plausible explanations and rank them by likelihood.
5. Highlight any inconsistencies where an event does not appear to explain the market reaction.
6. Consider macroeconomic, geopolitical, monetary policy, earnings, and sentiment-related factors when relevant.
7. Provide concise but insightful reasoning using professional financial terminology.

Output format:

## Market Summary
Brief overview of the day's market performance.

## Key Market Movements
- Movement:
- Likely Drivers:
- Explanation:

## Event Impact Analysis
For each event:
- Event:
- Affected Assets:
- Expected Impact:
- Observed Impact:
- Confidence Level (High/Medium/Low):
- Reasoning:

## Most Likely Market Drivers
Rank the top events that most likely drove the market.

## Conclusion
Summarize the primary reasons behind the market movements.

Language: English."""

    user_content = f"Stock: {stock_name}, Date: {date}, Price: {price}, News: {news_context}"

    try:
        completion=groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"
    
def plot_anomalies(data, stock_name):
    plt.figure(figsize=(12,6))
    plt.plot(data.index, data['Close'], label= 'Stock Price', color= 'blue', alpha=0.6)

    anomalies = data[data['Is_Anomaly'] == True]
    plt.scatter(anomalies.index, anomalies['Close'], color='red', label='Anomaly', s=100, zorder=5)
    plt.title(f"{stock_name} - Automated Anomaly Detection Dashboard", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price ($)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5) 
    plt.legend(loc='upper left', fontsize=11)
    plt.show()

def get_and_verify_date(data):
    user_date= input("Enter the anomaly date to analyze (YYYY-MM-DD): ")

    try:
        target_date= pd.to_datetime(user_date)
        if target_date in data.index:
            selected_row= data.loc[target_date]
            if selected_row['Is_Anomaly']:
                print("Anomaly verified on this date.")
            else:
                print("This date exists but isnt flagged as anomaly. But the internet will still be searched.")

        else:
            print("This date doesnt exist on the dataset (could be a weekend or holiday)")
            return None,None
        
    except Exception as e:
        print("Invalid date format. Please use YYYY-MM-DD")
        return None,None