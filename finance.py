import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
data = yf.download('RELIANCE.NS', start='2007-01-01', end='2023-10-10')
df_Open = data[['Open']]
df_dates = data.index.to_frame().reset_index(drop=True)
df_dates['Date'] = pd.to_datetime(df_dates['Date'])
df_dates['Date'] = df_dates['Date'].dt.strftime('%Y-%m-%d')
df_dates.set_index('Date', inplace=True)
X = data[['Open']]
y_close = data['Close']
y_high = data['High']
X_train, X_test, y_close_train, y_close_test, y_high_train, y_high_test = train_test_split(
    X, y_close, y_high, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model_close = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],),
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(1)
])

model_close.compile(optimizer='adam', loss='mean_squared_error')
history_close = model_close.fit(X_train_scaled, y_close_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
model_high = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_shape=(X_train_scaled.shape[1],),
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(1)
])
model_high.compile(optimizer='adam', loss='mean_squared_error')
history_high = model_high.fit(X_train_scaled, y_high_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
start_date = data.index[-1] - timedelta(days=6)
end_date = data.index[-1]
weekly_data = data[start_date:end_date]
daily_scaled = scaler.transform(weekly_data[['Open']])
daily_close_pred = model_close.predict(daily_scaled)
daily_high_pred = model_high.predict(daily_scaled)
weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
today_opening_price = float(input('Enter today\'s opening price: '))
today_opening_price_scaled = scaler.transform(np.array([[today_opening_price]]))
today_close_pred = model_close.predict(today_opening_price_scaled)
today_high_pred = model_high.predict(today_opening_price_scaled)
today_date = datetime.now().strftime('%Y-%m-%d')
x_labels = [f'{weekly_data.index[i].strftime("%Y-%m-%d")} ({weekdays[i]})' for i in range(len(weekdays))] + [f'{today_date} ({weekdays[0]})']
closing_prices = daily_close_pred.flatten().tolist() + [today_close_pred[0][0]]
high_prices = daily_high_pred.flatten().tolist() + [today_high_pred[0][0]]
plt.figure(figsize=(12, 6))
plt.plot(x_labels, weekly_data['Close'].tolist() + [today_opening_price], label='Actual Closing Price', marker='o', linestyle='-')
plt.plot(x_labels, daily_close_pred.flatten().tolist() + [today_close_pred[0][0]], label='Projected Closing Price', marker='x', linestyle='--')
plt.xlabel('Date (Day of the Week)')
plt.ylabel('Closing Price')
plt.title('Weekly Closing Price vs. Projected Closing Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(12, 6))
plt.plot(x_labels, weekly_data['High'].tolist() + [today_opening_price], label='Actual High Price', marker='o', linestyle='-')
plt.plot(x_labels, daily_high_pred.flatten().tolist() + [today_high_pred[0][0]], label='Projected High Price', marker='x', linestyle='--')
plt.xlabel('Date (Day of the Week)')
plt.ylabel('High Price')
plt.title('Weekly High Price vs. Projected High Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()
opening_prices = weekly_data['Open'].tolist() + [today_opening_price]
plt.figure(figsize=(12, 6))
plt.plot(x_labels, opening_prices, label='Opening Price', marker='o', linestyle='-')
plt.plot(x_labels, weekly_data['Close'].tolist() + [today_opening_price], label='Closing Price', marker='o', linestyle='-')
plt.xlabel('Date (Day of the Week)')
plt.ylabel('Price')
plt.title('Weekly Price Analysis with Today\'s Opening Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.show()