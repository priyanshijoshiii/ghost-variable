import pandas as pd

df = pd.read_csv('data/raw/compas-scores-two-years.csv')  ## reads the csv and turns into data frame
print(f"Raw Rows: {len(df)}")

## filter 1: keep rows where days_b_screening_arrest is between -30 to 30
df = df[(df['days_b_screening_arrest'] >= -30) & (df['days_b_screening_arrest'] <= 30)]
print(f"after screening length filter: {len(df)}")

print(f" recid_sum: {(df['is_recid'] == -1).sum()}")


## filter 2: drop calls where is_recid !=1
df = df[(df['is_recid'] != -1)]
print(f"after drop calls filter: {len(df)}")

## filter 3: drop c_charge_degree == 0

df = df[(df['c_charge_degree'] != 'O')]
print(f"after charge free filter: {len(df)}")

## filter 4: drop score_text == 'N/A'
df = df[(df['score_text'] != 'N/A')]
print(f"after score_charge filter {len(df)}")

df.to_csv('data/processed/compas_filtered.csv', index = False)
print("saved filtered dataset")