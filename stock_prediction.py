import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Set ticker and parameters
ticker = 'AAPL'
period = '2y'

print(f"--- Step 1: Fetching data for {ticker} ---")
data = yf.download(ticker, period=period)

if data.empty:
 print("Error: No data fetched. Check your internet connection or ticker symbol.")
 exit()

# Save original data to CSV
data.to_csv(f"{ticker}_stock_data.csv")
print(f"Data fetched and saved to '{ticker}_stock_data.csv' successfully. Total rows: {len(data)}")

# Create directory for plots
plots_dir = "plots"
if not os.path.exists(plots_dir):
 os.makedirs(plots_dir)

print("\n--- Step 2: Preprocessing Data ---")
# Use Open, High, Low, Close, Volume as features
# We want to predict the NEXT day's Close price
data['Next_Close'] = data['Close'].shift(-1)

# Inspect the last few rows to see the NaN in 'Next_Close'
print("Last 5 rows with 'Next_Close':")
print(data[['Close', 'Next_Close']].tail())

# Drop the last row as it has no 'Next_Close'
data_clean = data.dropna()

# Features (X) and Target (y)
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = data_clean[features]
y = data_clean['Next_Close']

# Split data - using a sequential split for time series
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"Training set: {len(X_train)} rows")
print(f"Testing set: {len(X_test)} rows")

print("\n--- Step 3: Training the Model ---")
model = LinearRegression()
model.fit(X_train, y_train)

print("\n--- Step 4: Making Predictions and Evaluating ---")
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"R-squared (R2) Score: {r2:.4f}")

print("\n--- Step 5: Visualizing Results ---")
plt.figure(figsize=(14, 7))
plt.plot(y_test.index, y_test.values, label='Actual Close Price', color='blue', linewidth=2)
plt.plot(y_test.index, y_pred, label='Predicted Close Price', color='red', linestyle='--', linewidth=2)

plt.title(f'{ticker} Stock Price Prediction (Actual vs Predicted)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plot_path = os.path.join(plots_dir, "stock_prediction_plot.png")
plt.savefig(plot_path)
plt.close()

print(f"Plot saved successfully at: {plot_path}")
print("Analysis complete.")
