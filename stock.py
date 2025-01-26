import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_stock_data(stock_name):
    stock = yf.Ticker(stock_name)
    data = stock.history(period="1y")  # Getting 1 year of stock data
    return data

def feature_engineering(data):
    data['Returns'] = data['Close'].pct_change()  # Daily returns
    data['Moving_Avg'] = data['Close'].rolling(window=5).mean()  # 5-day moving average
    data['Volatility'] = data['Returns'].rolling(window=5).std()  # 5-day volatility
    data['Direction'] = np.where(data['Returns'] > 0, 1, 0)  # 1 for bull, 0 for bear
    data.dropna(inplace=True)  # Drop NA values that may have been generated during calculation
    return data

def train_model(data):
    X = data[['Returns', 'Moving_Avg', 'Volatility']]
    y = data['Direction']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a RandomForest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Test the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
    
    return model

def predict_stock_trend(stock_name, model):
    data = get_stock_data(stock_name)
    data = feature_engineering(data)
    X = data[['Returns', 'Moving_Avg', 'Volatility']].iloc[-1:]
    prediction = model.predict(X)
    trend = 'Bull Market' if prediction == 1 else 'Bear Market'
    print(f'The stock {stock_name} is currently in a {trend}')

if __name__ == "__main__":
    stock_name = input("Enter the stock ticker symbol (e.g., AAPL, TSLA, MSFT): ").upper()
    
    # Get stock data and feature engineering
    data = get_stock_data(stock_name)
    data = feature_engineering(data)
    
    # Train the model
    model = train_model(data)
    
    # Predict the trend for the entered stock
    predict_stock_trend(stock_name, model)
