# ================== АВТОУСТАНОВКА БИБЛИОТЕК ==================

import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for package in ['tensorflow', 'keras', 'keras-tuner', 'numpy', 'pandas', 'matplotlib', 'scikit-learn']:
    try:
        __import__(package.replace("-", "_"))
    except ImportError:
        print(f"Устанавливаю {package}...")
        install(package)

# ================== ИМПОРТЫ ==================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Conv1D, MaxPooling1D, Flatten, Attention, Bidirectional, Input
from keras.optimizers import Adam
from keras import Model
import os

# ================== ЗАГРУЗКА ДАННЫХ ==================

RESULT_CSV1 = "nyt_2024_news_sentiment_deberta-v3-small_turbo.csv"
all_news_df = pd.read_csv(RESULT_CSV1, parse_dates=['published_date'])

RESULT_CSV2 = "financial_data.csv"

def read_financial_data_with_multiindex(csv_path):
    df = pd.read_csv(csv_path)
    date_column = df['Date']
    df_rest = df.drop(columns=['Date'])
    new_columns = df_rest.columns.str.split('.', expand=True)
    df_rest.columns = pd.MultiIndex.from_tuples(new_columns)
    df_rest['Date'] = pd.to_datetime(date_column)
    df_rest = df_rest.set_index('Date')
    return df_rest

financial_data = read_financial_data_with_multiindex(RESULT_CSV2)

# ================== ПОДГОТОВКА ДАННЫХ ==================

sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1}
all_news_df['sentiment_score'] = all_news_df['sentiment'].map(sentiment_map)

daily_sentiment = all_news_df.groupby('published_date').agg(
    avg_sentiment_score=('sentiment_score', 'mean'),
    news_count=('sentiment_score', 'count')
).reset_index()

open_data = financial_data.xs('Open', axis=1, level=1)
open_data.index = pd.to_datetime(open_data.index)
open_data = open_data.reset_index()

data = pd.merge(daily_sentiment, open_data, left_on='published_date', right_on='Date', how='inner')
data.drop(columns=['Date'], inplace=True)

tickers = open_data.columns.drop('Date')

# ================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==================

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step), :])
        y.append(dataset[i + time_step, 1])
    return np.array(X), np.array(y)

def build_enhanced_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    attention = Attention()([x, x])
    x = Flatten()(attention)
    x = Dense(50, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1)(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def plot_predictions(y_true, y_pred, ticker, mode):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Истинная цена")
    plt.plot(y_pred, label="Предсказанная", alpha=0.7)
    plt.title(f"{ticker}: Цена Open ({mode})")
    plt.xlabel("Время")
    plt.ylabel("Цена открытия")
    plt.legend()
    plt.grid(True)

    filename = f"plots/{ticker}_{mode}.png"
    plt.savefig(filename)
    plt.close()
    print(f"📀 Сохранен график: {filename}")

os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ================== ОБУЧЕНИЕ ==================

print("\n=== Статическое обучение (Fixed Time Step) ===")

STATIC_TIME_STEP = 5

for ticker in tickers:
    if ticker not in data.columns:
        continue

    print(f"\nОбучение для {ticker}...")

    ticker_data = data[['avg_sentiment_score', ticker]].dropna()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(ticker_data)

    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    X_train, y_train = create_dataset(train_data, STATIC_TIME_STEP)
    X_test, y_test = create_dataset(test_data, STATIC_TIME_STEP)

    if len(X_train) == 0 or len(X_test) == 0:
        print(f"⚠️ Недостаточно данных {ticker}")
        continue

    model = build_enhanced_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    model.save(f'models/lstm_model_static_{ticker}.keras')

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(np.concatenate((np.zeros((predictions.shape[0], 1)), predictions), axis=1))[:, 1]
    y_test_actual = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 1)), y_test.reshape(-1, 1)), axis=1))[:, 1]

    r2 = r2_score(y_test_actual, predictions)
    print(f"🔵 R2 Score для {ticker}: {r2:.4f}")

    plot_predictions(y_test_actual, predictions, ticker, mode="static")

print("\n=== Завершено! ===")
