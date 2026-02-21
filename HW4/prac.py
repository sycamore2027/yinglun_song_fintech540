import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

cin = pd.read_csv('yinglun_song_fintech540/HW4/test7_1.csv')

def historicalVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        return np.percentile(returns, alpha)
    
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalVaR, alpha=alpha)
    
    else:
        raise TypeError("Expected returns is neither df or Series")

def historicalCVaR(returns, alpha=5):
    if isinstance(returns, pd.Series):
        belowVaR = returns <= historicalVaR(returns, alpha)
        return returns[belowVaR].mean()
    elif isinstance(returns, pd.DataFrame):
        return returns.aggregate(historicalCVaR, alpha=alpha)
    else:
        raise TypeError("not expecte datatype")

print('VaR:  ', historicalVaR(cin, 5))
print('CVaR: ', historicalCVaR(cin, 5))
print(np.sqrt(len(cin)))
