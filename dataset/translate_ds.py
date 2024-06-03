import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from deep_translator import GoogleTranslator
from tqdm import tqdm

# Функция для разбивки текста на части не более 5000 символов
def split_text(text, max_length=5000):
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

# Функция для перевода текста с разбиением на части
def translate_text_deepL(text):
    try:
        parts = split_text(text)
        translated_parts = [GoogleTranslator(source='en', target='ru').translate(part) for part in parts]
        return ''.join(translated_parts)
    except Exception as e:
        print(f"Translation error: {e}")
        return text

# Получаем абсолютный путь к папке 'data' относительно текущего скрипта
current_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))

# Читаем train, val и test CSV файлы
train_df = pd.read_csv(os.path.join(data_dir, 'mental_disorders', 'train_renamed.csv'))
val_df = pd.read_csv(os.path.join(data_dir, 'mental_disorders', 'val_renamed.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'mental_disorders', 'test_renamed.csv'))

# Функция для перевода датасета с прогресс баром
def translate_dataset_with_progress(df):
    texts = df['text'].tolist()
    with ThreadPoolExecutor() as executor:
        translated_texts = list(tqdm(executor.map(translate_text_deepL, texts), total=len(texts)))
    df['text'] = translated_texts
    return df

# Переводим и сохраняем датасеты
train_df = translate_dataset_with_progress(train_df)
train_df.to_csv(os.path.join(data_dir, 'mental_disorders', 'train_translated.csv'), index=False)

val_df = translate_dataset_with_progress(val_df)
val_df.to_csv(os.path.join(data_dir, 'mental_disorders', 'val_translated.csv'), index=False)

test_df = translate_dataset_with_progress(test_df)
test_df.to_csv(os.path.join(data_dir, 'mental_disorders', 'test_translated.csv'), index=False)

print("Text in 'text' column has been translated using DeepL Translator and saved for train, val, and test datasets.")
