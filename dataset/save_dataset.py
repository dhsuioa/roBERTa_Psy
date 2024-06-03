import pandas as pd
import pyarrow.parquet as pq
import os

# Получаем абсолютный путь к папке 'data' относительно текущего скрипта
current_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))

# Создаем папку 'mental_disorders', если она не существует
output_dir = os.path.join(data_dir, 'mental_disorders')
os.makedirs(output_dir, exist_ok=True)

# Абсолютные пути к файлам Parquet
train_parquet_path = os.path.join(data_dir, 'train-00000-of-00001.parquet')
test_parquet_path = os.path.join(data_dir, 'test-00000-of-00001.parquet')
val_parquet_path = os.path.join(data_dir, 'val-00000-of-00001.parquet')

# Загрузка и преобразование данных в pandas DataFrame
train_dataset = pq.read_table(train_parquet_path)
df_train = train_dataset.to_pandas()

test_dataset = pq.read_table(test_parquet_path)
df_test = test_dataset.to_pandas()

val_dataset = pq.read_table(val_parquet_path)
df_val = val_dataset.to_pandas()

# Пример сохранения в формате CSV
df_train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
df_test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
df_val.to_csv(os.path.join(output_dir, 'val.csv'), index=False)

print("Datasets saved as CSV in 'data/mental_disorders'")
