import pandas as pd
import os

# Получаем абсолютный путь к папке 'data' относительно текущего скрипта
current_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))

# Читаем train, val и test CSV файлы
train_df = pd.read_csv(os.path.join(data_dir, 'mental_disorders', 'train.csv'))
val_df = pd.read_csv(os.path.join(data_dir, 'mental_disorders', 'val.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'mental_disorders', 'test.csv'))

# Создаем словарь для замены
label_mapping = {
    0: 'ПРЛ',
    1: 'Биполярное расстройство',
    2: 'Депрессия',
    3: 'Тревога',
    4: 'Шизофрения',
    5: 'Психическое заболевание'
}

# Заменяем значения в столбце label для train, val и test датасетов
train_df['label'] = train_df['label'].replace(label_mapping)
val_df['label'] = val_df['label'].replace(label_mapping)
test_df['label'] = test_df['label'].replace(label_mapping)

# Сохраняем измененные датасеты обратно в CSV файлы
train_df.to_csv(os.path.join(data_dir, 'mental_disorders', 'train_renamed.csv'), index=False)
val_df.to_csv(os.path.join(data_dir, 'mental_disorders', 'val_renamed.csv'), index=False)
test_df.to_csv(os.path.join(data_dir, 'mental_disorders', 'test_renamed.csv'), index=False)

print("Labels in 'label' column have been replaced and saved for train, val, and test datasets.")
