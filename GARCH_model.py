# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Download historical stock data
symbol = "TSLA"  # Replace with the desired stock symbol
data = yf.download(symbol, start="2020-01-01", end="2025-01-01")

# Print available column names and first few rows for debugging
print("Available columns:", data.columns)
print(data.head())

# Check if 'Close' exists
if "Close" in data.columns:
    price_column = "Close"
else:
    raise KeyError("'Close' column not found in data")

# Compute log returns
data["Log Returns"] = np.log(data[price_column] / data[price_column].shift(1))
data.dropna(inplace=True)

# Fit GARCH(1,1) model
model = arch_model(data["Log Returns"], vol="Garch", p=1, q=1)
result = model.fit(disp="off")

# Extract conditional volatility
volatility = result.conditional_volatility

data["Volatility"] = volatility

# Feature Engineering
# Adding lagged returns and moving average as features
data["Lagged_Return"] = data["Log Returns"].shift(1)
data["MA_5"] = data["Log Returns"].rolling(window=5).mean()
data.dropna(inplace=True)

# Prepare data for ML model
features = ["Lagged_Return", "MA_5"]
target = "Volatility"
X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Random Forest MSE: {mse}")

# Plot conditional volatility and ML predictions
plt.figure(figsize=(10,5))
plt.scatter(y_test.index, y_test, label="Actual Volatility", color='blue')
plt.scatter(y_test.index, y_pred, label="Predicted Volatility", color='red', linestyle='dashed')
plt.title(f"GARCH + ML Volatility Prediction for {symbol}")
plt.legend()
plt.show()
