import numpy as np
from numpy.linalg import inv

def lm_fs_fast(X_all, y, base_vars, use_const = True):
    n, p = X_all.shape
    if use_const:
        p -= 1
    X_base = X_all[:, base_vars]
    G_inv = inv(X_base.T @ X_base)
    b = X_base.T @ y
    total_var = np.sum((y - y.mean()) ** 2)

    r2s = {}
    base_set = set(base_vars)
    for j in range(p):
        if j in base_set:
            continue

        xj = X_all[:, j:j + 1]# (n,1)
        B = X_base.T @ xj
        C = xj.T @ X_base 
        D = xj.T @ xj

        # s = Schur complement
        S = D - (C @ G_inv @ B)
        S_inv = 1 / S
        
        if S[0][0] <= 1e-10:
            r2s[j] = np.nan  # ill-conditioned
            continue

        G_inv_new_lower = (C @ G_inv)
        G_inv_updated = np.vstack((G_inv + S_inv * G_inv_new_lower.T @ G_inv_new_lower, -S_inv * G_inv_new_lower))
        G_inv_updated = np.hstack((G_inv_updated, np.hstack([-G_inv_new_lower * S_inv, S_inv]).T))
        b_new = np.hstack((b, xj.T @ y))
        beta_new = G_inv_updated @ b_new    
        X_full = np.hstack([X_base, xj])
        y_pred = X_full @ beta_new
        ss_res = np.sum((y - y_pred.flatten()) ** 2)
        r2 = 1 - ss_res / total_var
        r2s[j] = r2
    return r2s

def lm_el_fast(X, y, use_const = True):
    n, p = X.shape
    ids = np.arange(p)
    if use_const:
        p -= 1
    G_inv = inv(X.T @ X)
    b = X.T @ y
    total_var = np.sum((y - y.mean()) ** 2)
    r2s = np.zeros(p)

    
    for j in range(p):
        idx = ids != j

        # Extract M, alpha, v from full G_inv
        M = G_inv[np.ix_(idx, idx)] # (p-1)x(p-1)
        v = G_inv[idx, j].reshape(-1, 1)# (p-1)x1
        alpha = G_inv[j, j]# scalar
        # Schur-complement-based inverse of G_sub
        G_sub_inv = M - (v @ v.T) / alpha

        # b_sub for beta
        b_sub = b[idx].reshape(-1, 1) # (p-1)x1
        beta_sub = (G_sub_inv @ b_sub).flatten()  # (p-1)x1

        # Predict y using X_{-j}
        y_pred = X[:, idx] @ beta_sub
        ss_res = np.sum((y - y_pred) ** 2)
        r2s[j] = 1 - ss_res / total_var

    return r2s