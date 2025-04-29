import yfinance as yf
import pandas as pd

tickers = [
    "NVDA",  # NVIDIA
    "INTC",  # Intel
    "AMD",   # AMD
    "JPM",   # JPMorgan Chase
    "GS",    # Goldman Sachs
    "BAC",   # Bank of America
    "DUK",   # Duke Energy
    "XOM",   # Exxon Mobil
    "CVX",   # Chevron
    "BA",    # Boeing
    "GD",    # General Dynamics
    "LMT"    # Lockheed Martin
]

def download_and_prepare_financial_data(tickers, start_date, end_date):
    # Скачиваем данные
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    
    # Проверяем, если мультииндекс
    if isinstance(data.columns, pd.MultiIndex):
        # Объединяем уровни мультииндекса в одно имя колонки (через точку)
        data.columns = [f"{ticker}.{field}" for ticker, field in data.columns]
    
    # Сбросим индекс чтобы дата стала обычной колонкой
    data = data.reset_index()
    
    return data

start_date = "2024-01-01"
end_date = "2025-01-01"

# Парсим и сразу готовим нормальный датафрейм
financial_data = download_and_prepare_financial_data(tickers, start_date, end_date)

# Сохраняем
financial_data.to_csv("financial_data.csv", index=False)

print("✅ Данные успешно загружены и сохранены в 'financial_data.csv'")
