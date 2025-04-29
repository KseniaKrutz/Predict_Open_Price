import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset, random_split

# Проверка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Работаем на устройстве: {device}")

# ================== ЗАГРУЗКА ДАННЫХ ==================

# ---------- 1. Загрузка новостей ----------
RESULT_CSV1 = "nyt_2024_news_sentiment_deberta-v3-small_turbo.csv"
all_news_df = pd.read_csv(RESULT_CSV1, parse_dates=['published_date'])

# ---------- 2. Загрузка финансовых данных ----------
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

# Маппинг sentiment в числовые значения
sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1}
all_news_df['sentiment_score'] = all_news_df['sentiment'].map(sentiment_map)

# Агрегация по дате
daily_sentiment = all_news_df.groupby('published_date').agg(
    avg_sentiment_score=('sentiment_score', 'mean'),
    news_count=('sentiment_score', 'count')
).reset_index()

# Берем только цену открытия (Open) всех тикеров
open_data = financial_data.xs('Open', axis=1, level=1)
open_data.index = pd.to_datetime(open_data.index)
open_data = open_data.reset_index()

# Объединяем по дате
data = pd.merge(daily_sentiment, open_data, left_on='published_date', right_on='Date', how='inner')
data.drop(columns=['Date'], inplace=True)

# Сохраняем тикеры
tickers = open_data.columns.drop('Date')  # Все тикеры из финансовых данных

# Создание необходимых папок
os.makedirs('plots', exist_ok=True)
os.makedirs('modelsTST', exist_ok=True)

# Класс модели Transformer
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_heads=4, num_layers=2, output_dim=1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim*2,
            activation='gelu'
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x, x)
        x = self.fc_out(x[:, -1, :])  # Используем последний таймстеп
        return x

# Функция создания датасета
def create_dataset(data, time_step=30):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, 1])  # Предсказание цены
    return np.array(X), np.array(y)

# Функция для обучения модели
def train_model(model, train_loader, epochs=20, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = loss_fn(output.squeeze(), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Функция для предсказания
def predict(model, loader):
    model.eval()
    pred_list, actual_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()  # shape (batch, 1)
            true = yb.numpy()              # shape (batch,)
            pred_list.append(out.reshape(-1))     # полифилл в 1D
            actual_list.append(true.reshape(-1))
    preds = np.concatenate(pred_list, axis=0)      # shape (N,)
    actuals = np.concatenate(actual_list, axis=0)  # shape (N,)
    return preds, actuals

# Функция визуализации
def plot_predictions(y_true, y_pred, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="True Volume")
    plt.plot(y_pred, label="Predicted Volume", alpha=0.7)
    plt.title(f"Volume Prediction using Transformer - {ticker}")
    plt.xlabel("Time")
    plt.ylabel("Volume")
    plt.legend()
    plt.grid()
    plot_path = f"plots/{ticker}_transformer_volume.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Сохранен график: {plot_path}")

# Основной код обработки всех тикеров
max_time_step = 30  # Максимальное окно


for ticker in tickers:
    print(f"\n=== Обработка тикера: {ticker} ===")

    if ticker not in data.columns:
        print(f"Тикер {ticker} отсутствует в данных!")
        continue

    company_data = data[['avg_sentiment_score', ticker]].rename(columns={ticker: 'Volume'})

    if len(company_data) < 40:  # Минимум данных
        print(f"Недостаточно данных для {ticker}")
        continue

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(company_data)

    dynamic_time_step = min(max_time_step, len(company_data) // 3)
    dynamic_time_step = max(dynamic_time_step, 3)  # Минимальное окно

    X, y = create_dataset(scaled_data, dynamic_time_step)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Инициализация модели
    model = TimeSeriesTransformer(
        input_dim=2, hidden_dim=64, num_heads=4, num_layers=2, output_dim=1
    ).to(device)

    # Обучение модели
    train_model(model, train_loader, epochs=20, lr=0.001)

    # Предсказания
    preds, actuals = predict(model, test_loader)

    # Обратное масштабирование
    preds_rescaled = scaler.inverse_transform(
    np.vstack((np.zeros_like(preds), preds)).T
    )[:, 1]
    actuals_rescaled = scaler.inverse_transform(
    np.vstack((np.zeros_like(actuals), actuals)).T
    )[:, 1]
    
    # Вычисление R²
    r2 = r2_score(actuals_rescaled, preds_rescaled)
    print(f"R² для {ticker}: {r2:.4f}")

    # Визуализация
    plot_predictions(actuals_rescaled, preds_rescaled, ticker)

    # Сохранение модели
    model_path = f"modelsTST/transformer_volume_{ticker}.keras"
    torch.save(model.state_dict(), model_path)
    print(f"Модель сохранена: {model_path}")

print("\n✅ Все тикеры обработаны.")
