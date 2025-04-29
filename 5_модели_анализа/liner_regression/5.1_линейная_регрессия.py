import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------- 1. Загрузка новостей ----------
RESULT_CSV1 = "nyt_2024_news_sentiment_deberta-v3-small_turbo.csv"
all_news_df = pd.read_csv(RESULT_CSV1, parse_dates=['published_date'])

# Проверим уникальные значения в столбце 'sentiment'
print("Уникальные значения в столбце 'sentiment':", all_news_df['sentiment'].unique())

# ---------- 2. Загрузка финансовых данных ----------
RESULT_CSV2 = "financial_data.csv"

def read_financial_data_with_multiindex(csv_path):
    df = pd.read_csv(csv_path)

    # Сохраним колонку Date отдельно
    date_column = df['Date']

    # Теперь удалим её перед обработкой заголовков
    df_rest = df.drop(columns=['Date'])

    # Расщепим остальные заголовки по '.'
    new_columns = df_rest.columns.str.split('.', expand=True)
    df_rest.columns = pd.MultiIndex.from_tuples(new_columns)

    # Вернем Date обратно
    df_rest['Date'] = pd.to_datetime(date_column)
    df_rest = df_rest.set_index('Date')
    
    return df_rest

# Чтение финансовых данных
financial_data = read_financial_data_with_multiindex(RESULT_CSV2)

# ---------- 3. Подготовка данных ----------
# Считаем количество новостей по каждому дню и тональности
daily_sentiment = all_news_df.groupby(['published_date', 'sentiment']).size().unstack(fill_value=0).reset_index()

# Преобразуем столбец даты в правильный формат
daily_sentiment['published_date'] = pd.to_datetime(daily_sentiment['published_date'])

# Проверим, что в данных есть столбцы для 'positive', 'negative' и 'neutral'
print("Структура данных после группировки:", daily_sentiment.head())

# ---------- 4. Подготовка цены открытия ----------
open_data = financial_data.xs('Open', axis=1, level=1)
open_data.index = pd.to_datetime(open_data.index)
open_data = open_data.reset_index()

# ---------- 5. Объединение по дате ----------
merged_data = pd.merge(daily_sentiment, open_data, left_on='published_date', right_on='Date', how='inner')

# Убираем столбец с датой из финальных данных
merged_data = merged_data.drop(columns=['Date'])

# ---------- 6. Сдвиг признаков на 1 и 2 дня вперед ----------
# Проверим, что в данных есть столбцы 'positive', 'negative' и 'neutral', если их нет, заменим на 0
if 'positive' not in merged_data.columns:
    merged_data['positive'] = 0
if 'negative' not in merged_data.columns:
    merged_data['negative'] = 0
if 'neutral' not in merged_data.columns:
    merged_data['neutral'] = 0

# Сдвигаем количество новостей на 1 и 2 дня вперед
merged_data['positive_lag1'] = merged_data['positive'].shift(1)
merged_data['negative_lag1'] = merged_data['negative'].shift(1)
merged_data['neutral_lag1'] = merged_data['neutral'].shift(1)

merged_data['positive_lag2'] = merged_data['positive'].shift(2)
merged_data['negative_lag2'] = merged_data['negative'].shift(2)
merged_data['neutral_lag2'] = merged_data['neutral'].shift(2)

# Убираем строки с пропусками (NaN)
merged_data = merged_data.dropna()

# ---------- 7. Линейная регрессия ----------
for ticker in open_data.columns[1:]:
    print(f"\n📈 Анализ для {ticker}:")

    y = merged_data[ticker]  # Целевая переменная (цена Open)

    X = merged_data[['positive_lag1', 'negative_lag1', 'neutral_lag1', 
                     'positive_lag2', 'negative_lag2', 'neutral_lag2']]  # Признаки

    # Создание и обучение модели линейной регрессии
    model = LinearRegression()
    model.fit(X, y)

    # Предсказания
    y_pred = model.predict(X)

    # Коэффициент детерминации R²
    r2 = r2_score(y, y_pred)
    print(f"R² = {r2:.3f}")

    # Коэффициенты модели
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    })
    print(coef_df)

    # Визуализация: Реальная и предсказанная цена открытия
    plt.figure(figsize=(10, 5))
    plt.plot(merged_data['published_date'], y, label='Real Open', color='blue')
    plt.plot(merged_data['published_date'], y_pred, label='Predicted open', color='red', linestyle='--')
    plt.title(f"{ticker} - Реальная и предсказанная цена открытия")
    plt.xlabel('Date')
    plt.ylabel('Open')
    plt.legend()
    plt.show()
