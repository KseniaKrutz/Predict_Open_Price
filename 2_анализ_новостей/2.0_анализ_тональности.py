# 2. Импорты
import os
import glob
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import DebertaV2Tokenizer, AutoModelForSequenceClassification
from collections import Counter

# 3. Настройки
SAVE_DIR = "nyt_2024_sections"
MODEL_NAME = "microsoft/deberta-v3-small"  # Можно заменить на любую другую
CHUNK_SIZE = 500  # Максимум токенов в чанке
BATCH_SIZE = 128  # Батч теперь огромный, потому что GPU будет справляться

# 4. Подключение к устройству
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Работаем на {device}")

# 5. Загружаем данные
csv_files = glob.glob(os.path.join(SAVE_DIR, "*.csv"))
df_list = [pd.read_csv(file) for file in csv_files]
all_news_df = pd.concat(df_list, ignore_index=True)

all_news_df['published_date'] = pd.to_datetime(all_news_df['published_date']).dt.date
assert 'full_text' in all_news_df.columns, "Ошибка: нет колонки 'full_text'."
print(f"✅ Загружено {len(all_news_df)} новостей.")

# 6. Загружаем токенизатор и модель
print(f"🧠 Загружаем модель {MODEL_NAME}...")
tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to(device)
model.eval() # Переводим в режим инференса

# 7. Функция деления текста на чанки
def chunk_text_by_tokens(text, tokenizer, max_tokens=500):
    if pd.isnull(text) or not text.strip():
        return []
    tokens = tokenizer(text, return_tensors=None, add_special_tokens=False)["input_ids"]
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_ids = tokens[i:i+max_tokens]
        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk)
    return chunks

# 8. Готовим все чанки
print("✂️ Режем новости на чанки...")
all_chunks = []
chunk_to_article_idx = []

for idx, text in tqdm(enumerate(all_news_df['full_text']), total=len(all_news_df)):
    chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=CHUNK_SIZE)
    all_chunks.extend(chunks)
    chunk_to_article_idx.extend([idx] * len(chunks))

print(f"✅ Получено {len(all_chunks)} чанков.")

# 9. Создаем датасет и загрузчик
class TextChunksDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {key: val.squeeze(0) for key, val in encoding.items()}

dataset = TextChunksDataset(all_chunks, tokenizer)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# 10. Прогоняем через модель
print("🚀 Анализируем все чанки через DataLoader...")
all_preds = []

with torch.no_grad():
    for batch in tqdm(dataloader):
        batch = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**batch)
        probs = outputs.logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)
        all_preds.extend(preds.cpu().tolist())

# 11. Превращаем предсказания в текстовые лейблы
id2label = {0: "negative", 1: "neutral", 2: "positive"}
chunk_sentiments = [id2label.get(pred, "neutral") for pred in all_preds]

# 12. Агрегация результатов
print("🔧 Агрегируем результаты...")

article_sentiments = [[] for _ in range(len(all_news_df))]
for idx, sentiment in zip(chunk_to_article_idx, chunk_sentiments):
    article_sentiments[idx].append(sentiment)

final_sentiments = []
for sentiments in article_sentiments:
    if not sentiments:
        final_sentiments.append('neutral')
        continue
    counter = Counter(sentiments)
    most_common = counter.most_common(1)[0][0]
    final_sentiments.append(most_common)

all_news_df['sentiment'] = final_sentiments

# 13. Сохраняем результат
output_file = f"nyt_2024_news_sentiment_{MODEL_NAME.split('/')[-1]}_turbo.csv"
all_news_df[['published_date', 'section', 'title', 'url', 'sentiment', 'full_text']].to_csv(output_file, index=False)

print(f"✅ Готово! Результат сохранён в {output_file}.")
