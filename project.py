#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 09:41:14 2023

@author: macintosh
"""

import pandas as pd
import matplotlib.pyplot as plt


tesla_df = pd.read_csv('./Tesla.csv - Tesla.csv.csv')
hyundai_df = pd.read_csv('./005380.KS.csv')

#hyundai_data = pd.read_csv('./datasets/archive/005380.KS.csv')
#tesla_data = pd.read_csv('./datasets/Tesla.csv - Tesla.csv.csv')



# Display the first few rows of each dataset
print("Tesla Stock Prices:")
print(tesla_df.head())
print("\nHyundai Stock Prices:")
print(hyundai_df.head())

# Check the data types of each column
print("\nData Types:")
print(tesla_df.dtypes)
print(hyundai_df.dtypes)



# Plotting Tesla stock prices
plt.figure(figsize=(12, 6))
plt.plot(tesla_df['Date'], tesla_df['Close'], label='Tesla')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Tesla Stock Prices')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Plotting Hyundai stock prices
plt.figure(figsize=(12, 6))
plt.plot(hyundai_df['Date'], hyundai_df['Close'], label='Hyundai')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Hyundai Stock Prices')
plt.legend()
plt.xticks(rotation=45)
plt.show()




from sklearn.linear_model import LinearRegression

# Splitting the data into training and testing sets
train_data = tesla_df[['Close']].iloc[:-30]  # Use all but the last 30 days for training
test_data = tesla_df[['Close']].iloc[-30:]  # Use the last 30 days for testing

# Creating the linear regression model
model = LinearRegression()

# Training the model
model.fit(train_data.index.values.reshape(-1, 1), train_data['Close'])

# Predicting the closing prices for the test data
predictions = model.predict(test_data.index.values.reshape(-1, 1))

# Visualizing the predictions
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data['Close'], label='Actual')
plt.plot(test_data.index, predictions, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Tesla Stock Price Prediction')
plt.legend()
plt.xticks(rotation=45)
plt.show()