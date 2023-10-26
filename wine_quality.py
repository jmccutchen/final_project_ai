import numpy as np
import pandas as pd

df = pd.concat(map(pd.read_csv, ['winequality-red.csv', 'winequality-white.csv']),ignore_index=True)
print(df.head())

# print dataset size
print(df.shape)

