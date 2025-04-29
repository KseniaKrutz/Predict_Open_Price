# 2.2_ключевые_слова.py

import nltk
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# 1. Константы
RESULT_CSV = "nyt_2024_news_sentiment_deberta-v3-small_turbo.csv"

# 2. Чтение данных
all_news_df = pd.read_csv(RESULT_CSV, parse_dates=['published_date'])

# 3. Нормализация sentiment
all_news_df['sentiment'] = all_news_df['sentiment'].astype(str).str.strip().str.lower()

# 4. Загрузка стоп-слов
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# 5. Предобработка текста
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    words = text.lower().split()
    words = [w for w in words if w.isalpha() and w not in stop_words]
    return ' '.join(words)

all_news_df['clean_text'] = all_news_df['full_text'].apply(preprocess_text)

# 6. Функция биграмм
def get_top_bigrams(texts, n=30):
    texts = texts.dropna().str.strip()
    texts = texts[texts != '']
    if texts.empty:
        return []
    vec = CountVectorizer(ngram_range=(2,2), stop_words='english')
    X = vec.fit_transform(texts)
    if X.shape[1] == 0:
        return []
    counts = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    return sorted(zip(terms, counts), key=lambda x: x[1], reverse=True)[:n]

def safe_get_top_bigrams(texts, label):
    cnt = texts.dropna().str.strip().shape[0]
    if cnt == 0:
        print(f"⚠️ Нет текстов для {label} ({cnt} записей).")
        return []
    return get_top_bigrams(texts)

# 7. Разделение по тональности
positive = all_news_df.loc[all_news_df['sentiment']=='positive', 'clean_text']
neutral  = all_news_df.loc[all_news_df['sentiment']=='neutral',  'clean_text']
negative = all_news_df.loc[all_news_df['sentiment']=='negative', 'clean_text']

# 8. Получаем топ-10 биграмм каждого
top_pos = safe_get_top_bigrams(positive, "positive")[:10]
top_neu = safe_get_top_bigrams(neutral,  "neutral") [:10]
top_neg = safe_get_top_bigrams(negative, "negative")[:10]

print("🔥 Топ-10 позитивных биграмм:", top_pos)
print("😐 Топ-10 нейтральных биграмм:", top_neu)
print("💀 Топ-10 негативных биграмм:", top_neg)

# 9. Построение WordCloud
def generate_wordcloud(bigrams, title):
    if not bigrams:
        print(f"⚠️ Нет биграмм для {title}.")
        return
    text = ' '.join([t for t,_ in bigrams])
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f'WordCloud: {title}')
    plt.axis('off')
    plt.show()

generate_wordcloud(top_pos, "Positive News")
generate_wordcloud(top_neu, "Neutral News")
generate_wordcloud(top_neg, "Negative News")
