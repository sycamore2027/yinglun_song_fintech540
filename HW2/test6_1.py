import pandas as pd

def cal_arithmetic_return(prices):
    returns = prices.pct_change().dropna()
    return returns

cin = pd.read_csv("HW2/test6.csv")  
prices = cin.iloc[:, 1:].astype(float)
returns = cal_arithmetic_return(prices)
returns.insert(0, "Date", cin.iloc[1:, 0])
returns.columns = cin.columns
returns.to_csv("HW2/test6_1_output.csv", float_format="%.12g", index=False)