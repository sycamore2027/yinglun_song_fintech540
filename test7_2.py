import pandas as pd
from scipy import stats

data = pd.read_csv("/Users/alansong/Documents/Duke/FINTECH 545/FinTech-545-Fall2025/testfiles/data/test7_2.csv")["x1"]

nu, mu, sigma = stats.t.fit(data)

print("mu:", mu)
print("sigma:", sigma)
print("nu:", nu)
