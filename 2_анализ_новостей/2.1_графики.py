# 2.1_графики.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ─── 1. ЗАГРУЗКА ДАННЫХ ────────────────────────────────────────────────────────────

# Укажи тот же путь и имя CSV, в который ты сохранял результаты анализа тональности
RESULT_CSV = "nyt_2024_news_sentiment_deberta-v3-small_turbo.csv"

# Читаем DataFrame
all_news_df = pd.read_csv(RESULT_CSV, parse_dates=['published_date'])

# ─── 2. ПРЕОБРАБОТКА ─────────────────────────────────────────────────────────────

# Карта для перевода лейблов в числовые оценки
sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
all_news_df['sentiment_score'] = all_news_df['sentiment'].map(sentiment_map)

# Добавим колонки года и недели (ISO)
all_news_df['year'] = all_news_df['published_date'].dt.isocalendar().year
all_news_df['week'] = all_news_df['published_date'].dt.isocalendar().week
all_news_df['year_month'] = all_news_df['published_date'].dt.to_period('M')

# ─── 3. ГРАФИК ТРЕНДА ПО ДНЯМ ────────────────────────────────────────────────────

# Группируем данные по месяцам и считаем среднее значение sentiment_score
monthly_sentiment = all_news_df.groupby('year_month')['sentiment_score'].mean().reset_index()

# Строим график тренда
plt.figure(figsize=(15,5))
plt.plot(
    monthly_sentiment['year_month'].astype(str),
    monthly_sentiment['sentiment_score'],
    marker='o', linestyle='-', alpha=0.7
)

# Добавляем скользящее среднее для сглаживания тренда (например, на 3 месяца)
monthly_sentiment['smoothed'] = monthly_sentiment['sentiment_score'].rolling(window=3).mean()

# Строим сглаженную линию
plt.plot(
    monthly_sentiment['year_month'].astype(str),
    monthly_sentiment['smoothed'],
    marker='', linestyle='--', color='red', label='Сглаженная линия (3 месяца)'
)

plt.title('Тональность новостей по месяцам (сглаженный тренд)')
plt.xlabel('Дата (месяц)')
plt.ylabel('Тональность: +1=позитив, 0=нейтрал, -1=негатив')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('nyt_sentiment_trend_smoothed.png')
plt.show()

print("✅ График тренда сохранён в 'nyt_sentiment_trend_smoothed.png'.")

# ─── 4. КРУГОВАЯ ДИАГРАММА РАСПРЕДЕЛЕНИЯ ──────────────────────────────────────

sentiment_counts = all_news_df['sentiment'].value_counts()
plt.figure(figsize=(7,7))
sentiment_counts.plot.pie(
    autopct='%1.1f%%',
    startangle=140,
    colors=["#66bb6a", "#ffa726", "#ef5350"]
)
plt.title('Распределение тональностей новостей')
plt.ylabel('')
plt.tight_layout()
plt.savefig('nyt_sentiment_piechart.png')
plt.show()
print("✅ Диаграмма распределения сохранена в 'nyt_sentiment_piechart.png'.")

# ─── 5. HEATMAP ПО НЕДЕЛЯМ ───────────────────────────────────────────────────────

# Сгруппируем по году и неделе и посчитаем средний sentiment_score
weekly_sentiment = (
    all_news_df
    .groupby(['year', 'week'])['sentiment_score']
    .mean()
    .reset_index()
)

# Пивотируем для тепловой карты
heatmap_data = weekly_sentiment.pivot(
    index='year',
    columns='week',
    values='sentiment_score'
)

plt.figure(figsize=(20, 6))
sns.heatmap(
    heatmap_data,
    cmap="RdYlGn",
    center=0,
    linewidths=0.5,
    annot=True,
    fmt=".2f"
)
plt.title('Heatmap тональности по неделям')
plt.xlabel('Неделя года')
plt.ylabel('Год')
plt.tight_layout()
plt.savefig('nyt_sentiment_heatmap.png')
plt.show()
print("✅ Heatmap сохранён в 'nyt_sentiment_heatmap.png'.")
