import pandas as pd
import numpy as np

def missing_val_cal(x, skipMiss = True, func = 'corr'):
    
    rows_with_empty = x.index[x.isna().any(axis=1)].tolist()

    if skipMiss: 
        for r in rows_with_empty:
            x = x.drop(r)
        if func == 'cov':
            answer =  x.cov()   
        elif func == 'corr':
            answer = x.corr() 
    else:
        columns = x.columns        
        n_cols = len(columns)
        data = x.values         
        out = np.zeros((n_cols, n_cols), dtype=float)        
        for i in range(n_cols):
            for j in range(i + 1):               
                col_i = data[:, i]                
                col_j = data[:, j]
                valid_mask = ~np.isnan(col_i) & ~np.isnan(col_j)
                valid_i = col_i[valid_mask]                
                valid_j = col_j[valid_mask]                
                val = np.nan
                if len(valid_i) > 1:                    
                    if func == 'cov':
                        val = np.cov(valid_i, valid_j)[0, 1]                    
                    elif func == 'corr':
                        val = np.corrcoef(valid_i, valid_j)[0, 1]
                out[i, j] = val                
                if i != j:                    
                    out[j, i] = val
        answer = pd.DataFrame(out, index=columns, columns=columns)
    return answer

if __name__ == "__main__":
    input_csv = pd.read_csv('HW2/test1.csv')
    output_csv = missing_val_cal(input_csv, skipMiss=False, func='corr')
    output_csv.to_csv('HW2/test1_4_output.csv', index=False)
    