import numpy as np
import pandas as pd
    
def rebonato_jackel(matrix, tol = 1e-15):
    matrix = np.asarray(matrix, dtype = float)
    
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input must be square matrix")
    
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals_clipped = np.maximum(eigvals, tol)
    lambda_p = np.diag(eigvals_clipped)
    
    S = eigvecs
    row_variances  =(S**2) @ eigvals_clipped
    row_variances = np.maximum(row_variances, tol)
    
    t = 1.0  / np.sqrt(row_variances)
    T = np.diag(t)
    
    C_hat = T @ S @ lambda_p @ S.T @ T
    

    return C_hat

cin = pd.read_csv('HW2/test1_4_output.csv')
cols = cin.columns.copy()

corr = cin.values.astype(float)

corr_fixed = rebonato_jackel(corr)

df_out = pd.DataFrame(corr_fixed)
df_out.to_csv('HW2/test3_2_output.csv', float_format="%.12g", index=False)