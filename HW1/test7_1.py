import pandas as pd

df = pd.read_csv("yinglun_song_fintech540/HW1/test7_1.csv")

mu = df["x1"].mean()
sigma = df["x1"].std()

print("mu:", mu)
print("sigma:", sigma)
