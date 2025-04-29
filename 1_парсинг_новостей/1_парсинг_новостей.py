import yfinance as yf
import pandas as pd
import feedparser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime
import glob
import os
from transformers import pipeline
from tqdm import tqdm
import requests
from time import sleep
from concurrent.futures import ThreadPoolExecutor
import nltk
nltk.download('stopwords')
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from bs4 import BeautifulSoup


API_KEY = "xnNtxALeCPPAwACO8C2EkAs7DvGBXte5"
SAVE_DIR = "nyt_2024_sections"
os.makedirs(SAVE_DIR, exist_ok=True)

def get_full_text(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Найдем все параграфы статьи
            paragraphs = soup.find_all('p')
            full_text = "\n".join(p.get_text() for p in paragraphs)
            return full_text
    except Exception as e:
        print(f"Ошибка при загрузке полного текста: {e}")
    return None

def download_month(month):
    url = f"https://api.nytimes.com/svc/archive/v1/2024/{month}.json?api-key={API_KEY}"
    print(f"📥 Загружаем {month:02d}/2024...")
    response = requests.get(url)

    month_articles = []

    if response.status_code == 200:
        data = response.json()
        docs = data['response']['docs']
        for doc in docs:
            article_url = doc.get('web_url')
            full_text = get_full_text(article_url) if article_url else None

            month_articles.append({
                "title": doc.get('headline', {}).get('main'),
                "url": article_url,
                "published_date": doc.get('pub_date'),
                "section": doc.get('section_name') or 'Unknown',
                "snippet": doc.get('snippet'),
                "source": doc.get('source'),
                "full_text": full_text
            })
    else:
        print(f"⚠️ Ошибка загрузки месяца {month}: {response.status_code}")

    return month_articles

def save_articles_by_section(all_articles):
    df = pd.DataFrame(all_articles)

    grouped = df.groupby('section')

    for section, group in grouped:
        safe_section = "".join(c if c.isalnum() else "_" for c in (section or "Unknown"))
        filename = os.path.join(SAVE_DIR, f"{safe_section}.csv")
        group.to_csv(filename, index=False)
        print(f"✅ Сохранено {len(group)} новостей в раздел '{section}'")

def main():
    all_articles = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(download_month, month) for month in range(1, 13)]
        for future in futures:
            all_articles.extend(future.result())

    print(f"📊 Всего собрано {len(all_articles)} новостей.")
    save_articles_by_section(all_articles)

if __name__ == "__main__":
    main()