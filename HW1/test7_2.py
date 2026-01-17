import pandas as pd
from scipy import stats

data = pd.read_csv("yinglun_song_fintech540/HW1/test7_2.csv")["x1"]

nu, mu, sigma = stats.t.fit(data)

print("mu:", mu)
print("sigma:", sigma)
print("nu:", nu)
