import pandas as pd
import numpy as np

def missing_val_cal(x, skipMiss = True, func = 'cov'):
    
    rows_with_empty = x.index[x.isna().any(axis=1)].tolist()

    if skipMiss: 
        for r in rows_with_empty:
            x = x.drop(r)
    if func == 'cov':
        answer =  x.cov()      
    return answer

if __name__ == "__main__":
    input_csv = pd.read_csv('HW2/test1.csv')
    output_csv = missing_val_cal(input_csv, skipMiss=True, func='cov')
    output_csv.to_csv('HW2/test1_1_output.csv', index=False)
    