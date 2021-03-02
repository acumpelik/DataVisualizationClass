import pandas as pd

df = pd.read_csv("C_DSSC_201_S21_Dataset1.csv", sep="'")

col = df.iloc[:, 0]
# print(col)
# df.shape
df.head()