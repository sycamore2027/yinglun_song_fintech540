import pandas as pd

df = pd.read_csv("FinTech-545-Fall2025/testfiles/data/test7_1.csv")

mu = df["x1"].mean()
sigma = df["x1"].std()

print("mu:", mu)
print("sigma:", sigma)
