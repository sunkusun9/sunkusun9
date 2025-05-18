import numpy as np
import pandas as pd
from numpy.linalg import inv
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def lm_fs_fast(X_all, y, base_vars, use_const = True, metric = r2_score):
    n, p = X_all.shape
    if use_const:
        p -= 1
    X_base = X_all[:, base_vars]
    G_inv = inv(X_base.T @ X_base)
    b = X_base.T @ y

    metrics = {}
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
        metrics[j] = metric(y, y_pred.flatten())
    return metrics

def lm_el_fast(X, y, use_const = True, metric = r2_score):
    n, p = X.shape
    ids = np.arange(p)
    if use_const:
        p -= 1
    G_inv = inv(X.T @ X)
    b = X.T @ y
    total_var = np.sum((y - y.mean()) ** 2)
    metrics = np.zeros(p)
    
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
        metrics[j] = metric(y, y_pred)

    return metrics

def lm_fs_fast2(X_all, y, base_vars, use_const = True, train_size = 0.8, random_state = 123, metric = r2_score):
    n, p = X_all.shape
    X_train, X_test, y_train, y_test = train_test_split(X_all, y, train_size = train_size, random_state = random_state)
    if use_const:
        p -= 1
        base_vars.append(-1)
    X_base = X_train[:, base_vars]
    G_inv = inv(X_base.T @ X_base)
    b = X_base.T @ y_train
    
    metrics = {}
    base_set = set(base_vars)
    for j in range(p):
        if j in base_set:
            continue

        xj = X_train[:, j:j + 1]# (n,1)
        B = X_base.T @ xj
        C = xj.T @ X_base 
        D = xj.T @ xj

        # s = Schur complement
        G_inv_new = (C @ G_inv)
        S = D - (G_inv_new @ B)
        
        if S[0][0] <= 1e-10:
            metric[j] = np.inf
            continue

        G_inv_new_lower = -G_inv_new / S
        G_inv_updated = np.block([
            [G_inv + G_inv_new.T @ G_inv_new / S, G_inv_new_lower.T],
            [G_inv_new_lower, 1 / S]
        ])
        b_new = np.hstack((b, xj.T @ y_train))
        beta_new = (G_inv_updated @ b_new).flatten()
        X_full = np.hstack([X_test[:, base_vars], X_test[:, j:j + 1]])
        y_pred = X_full @ beta_new
        metrics[j] = metric(y_test, y_pred)
    return metrics

def lm_el_fast2(X, y, use_const = True, train_size = 0.8, random_state = 123, metric = r2_score):
    n, p = X.shape
    ids = np.arange(p)
    if use_const:
        p -= 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size, random_state = random_state)
    G_inv = inv(X_train.T @ X_train)
    b = X_train.T @ y_train
    metrics = np.zeros(p)

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
        y_pred = X_test[:, idx] @ beta_sub
        metrics[j] = metric(y_test, y_pred)
    return metrics

def fs_fast(df, X, y, X_ff, metric=r2_score):
    return pd.Series(
        lm_fs_fast2(df[X_ff + ['const']].values, df[y], [X_ff.index(i) for i in X], metric=metric).values(),
        index = [i for i in X_ff if i not in X]
    ).sort_values()

def be_fast(df, X, y, floated_list, metric=r2_score):
    if len(X) < 2: return pd.Series()
    return pd.Series(
        lm_el_fast2(df[X + ['const']].values, df[y], metric=metric), index = X
    ).pipe(lambda x: x.loc[~x.index.isin(floated_list)]).sort_values()

def step_fs_fast(df, X, y, X_selected, floated_list, metric_list, metric=r2_score):
    while(True):
        s_metric = fs_fast(df, X_selected, y, X, metric)
        if len(s_metric) == 0:
            break
        if metric_list[-1] > s_metric.iloc[0]:
            metric_list.append(s_metric.iloc[0])
            X_selected.append(s_metric.index[0])
            #print("Selected: {}, RMSE: {}".format(X_selected[-1], metric_list[-1]))
        else:
            break
        while(True):
            s_metric = be_fast(df, X_selected, y, floated_list, metric)
            if len(s_metric) > 0 and metric_list[-1] > s_metric.iloc[0]:
                metric_list.append(s_metric.iloc[0])
                floated_list.add(s_metric.index[0])
                X_selected = [i for i in X_selected if i != s_metric.index[0]]
                #print("Excluded: {}, RMSE: {}".format(s_metric.index[0], metric_list[-1]))
            else:
                break
    return X_selected, floated_list, metric_list