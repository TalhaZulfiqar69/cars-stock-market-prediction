import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the datasets
tesla_df = pd.read_csv('./datasets/Tesla.csv - Tesla.csv.csv')
hyundai_df = pd.read_csv('./datasets/005380.KS.csv')

# Explore the data
print("Tesla Stock Prices:")
print(tesla_df.head())
print("\nHyundai Stock Prices:")
print(hyundai_df.head())

# Visualize the stock prices
plt.figure(figsize=(12, 6))
plt.plot(tesla_df['Date'], tesla_df['Close'], label='Tesla')
plt.plot(hyundai_df['Date'], hyundai_df['Close'], label='Hyundai')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Tesla vs Hyundai Stock Prices')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Merge the datasets for comparison
merged_df = pd.merge(tesla_df, hyundai_df, on='Date', suffixes=('_tesla', '_hyundai'))

# Prepare the data for neural network training
data = merged_df[['Close_tesla', 'Close_hyundai']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Split the data into training and testing sets
train_data = scaled_data[:-31]
test_data = scaled_data[-31:]

# Define the features and labels
X_train, y_train = train_data[:-1], train_data[1:]

# Reshape the input data for LSTM
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

# Build the LSTM model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(1, 2)))
model.add(Dense(2))
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, verbose=0)

# Extract the columns for visualization
columns_to_plot = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
tesla_data = tesla_df[columns_to_plot]
hyundai_data = hyundai_df[columns_to_plot]

# Visualize the data for Tesla
plt.figure(figsize=(12, 6))
for column in columns_to_plot:
    plt.plot(tesla_df['Date'], tesla_data[column], label=column)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Tesla Stock Data')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Visualize the data for Hyundai
plt.figure(figsize=(12, 6))
for column in columns_to_plot:
    plt.plot(hyundai_df['Date'], hyundai_data[column], label=column)
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Hyundai Stock Data')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Make predictions with the model
X_test = test_data[:-1]
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
predicted_data = model.predict(X_test)
predicted_data = scaler.inverse_transform(predicted_data)

# Compare the datasets and predicted data
plt.figure(figsize=(12, 6))
plt.plot(merged_df['Date'][-30:], merged_df['Close_tesla'][-30:], label='Actual Tesla')
plt.plot(merged_df['Date'][-30:], merged_df['Close_hyundai'][-30:], label='Actual Hyundai')
plt.plot(merged_df['Date'][-30:], predicted_data[:, 0], label='Predicted Tesla')
plt.plot(merged_df['Date'][-30:], predicted_data[:, 1], label='Predicted Hyundai')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Comparison between Actual and Predicted Stock Prices')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Calculate prediction errors
actual_tesla = merged_df['Close_tesla'][-30:].values
actual_hyundai = merged_df['Close_hyundai'][-30:].values
prediction_tesla = predicted_data[:, 0]
prediction_hyundai = predicted_data[:, 1]

# Calculate error metrics
mse_tesla = mean_squared_error(actual_tesla, prediction_tesla)
mse_hyundai = mean_squared_error(actual_hyundai, prediction_hyundai)
mae_tesla = mean_absolute_error(actual_tesla, prediction_tesla)
mae_hyundai = mean_absolute_error(actual_hyundai, prediction_hyundai)

# Visualize the prediction errors
plt.figure(figsize=(12, 6))
plt.plot(merged_df['Date'][-30:], actual_tesla - prediction_tesla, label='Tesla Error')
plt.plot(merged_df['Date'][-30:], actual_hyundai - prediction_hyundai, label='Hyundai Error')
plt.xlabel('Date')
plt.ylabel('Error')
plt.title('Prediction Errors')
plt.legend()
plt.xticks(rotation=45)
plt.show()

print("Mean Squared Error (Tesla):", mse_tesla)
print("Mean Absolute Error (Tesla):", mae_tesla)
print("Mean Squared Error (Hyundai):", mse_hyundai)
print("Mean Absolute Error (Hyundai):", mae_hyundai)
