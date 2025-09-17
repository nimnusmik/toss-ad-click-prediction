import pandas as pd
import polars as pl
import os
import pyarrow.parquet as pq

# Parquet 파일 열기
#table = pq.ParquetFile("/Users/sunminkim/Desktop/TOSS_ML/data/raw/train.parquet")

#os.makedirs("../data/divided/", exist_ok=True)

# 배치 단위로 읽고 CSV로 저장
#for i, batch in enumerate(table.iter_batches(batch_size=2000000)):
#    df = batch.to_pandas()  # Pandas DataFrame으로 변환
#    df.to_csv(f"../data/divided/train_batch_{i}.csv", index=False)  

file_path = '/Users/sunminkim/Desktop/TOSS_ML/data/raw/train.parquet'
table = pq.read_table(file_path)   
df = table.to_pandas()

sample_fraction = 0.2 #20% 샘플링 

stratified_sample_df = df.groupby('clicked', group_keys = False).apply(
    lambda x: x.sample(frac=sample_fraction, replace = False, random_state=42)
)

print(df['clicked'].value_counts(normalize=True))

stratified_sample_df.to_csv(f"../data/raw/train_sample.csv", index=False)
