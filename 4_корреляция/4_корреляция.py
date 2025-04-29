import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ---------- 1. Загрузка новостей ----------
RESULT_CSV1 = "nyt_2024_news_sentiment_deberta-v3-small_turbo.csv"
all_news_df = pd.read_csv(RESULT_CSV1, parse_dates=['published_date'])

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

# Берем только цену открытия (Open) всех тикеров
open_data = financial_data.xs('Open', axis=1, level=1)

# Убедимся, что индекс — это дата
open_data.index = pd.to_datetime(open_data.index)

# Сбросим индекс, чтобы дата стала обычной колонкой
open_data = open_data.reset_index()

# ---------- 4. Объединение по дате ----------
# Объединяем данные по дате
merged_data = pd.merge(daily_sentiment, open_data, left_on='published_date', right_on='Date', how='inner')

# Убираем колонки с датами перед корреляцией
features_for_corr = merged_data.drop(columns=['published_date', 'Date'])

# ---------- 5. Очистка высоко коррелированных признаков (если нужно) ----------
def remove_highly_correlated_features(corr_matrix, threshold=0.95):
    # Находим пары признаков с корреляцией выше threshold
    correlated_pairs = set()
    for i in range(corr_matrix.shape[0]):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname1 = corr_matrix.columns[i]
                colname2 = corr_matrix.columns[j]
                correlated_pairs.add(colname1)
    return list(correlated_pairs)

# Получаем корреляционную матрицу
correlation_matrix = features_for_corr.corr()

# Убираем высоко коррелированные признаки
high_corr_features = remove_highly_correlated_features(correlation_matrix, threshold=0.95)

if high_corr_features:
    print(f"⚡ Удаляем высоко коррелированные признаки: {high_corr_features}")
    features_for_corr = features_for_corr.drop(columns=high_corr_features)
    correlation_matrix = features_for_corr.corr()

# ---------- 6. ТОП-10 самых сильных корреляций ----------
corr_pairs = correlation_matrix.unstack().reset_index()
corr_pairs.columns = ['Feature1', 'Feature2', 'Correlation']

# Убираем дубликаты (где Feature1 == Feature2 и зеркальные пары)
corr_pairs = corr_pairs[corr_pairs['Feature1'] != corr_pairs['Feature2']]
corr_pairs['AbsCorrelation'] = corr_pairs['Correlation'].abs()
corr_pairs = corr_pairs.sort_values(by='AbsCorrelation', ascending=False).drop_duplicates(subset=['AbsCorrelation'])

# Печатаем ТОП-10 корреляций
print("\n🔥 ТОП-10 самых сильных корреляций:")
print(corr_pairs.head(10)[['Feature1', 'Feature2', 'Correlation']])

# ---------- 7. Визуализация ----------
plt.figure(figsize=(16,12))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Корреляция между количеством новостей и ценами открытия акций по дням", fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('correlation_map.png')
plt.show()

print("✅ Heatmap сохранён в 'correlation_map.png'.")
