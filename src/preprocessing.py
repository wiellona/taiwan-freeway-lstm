import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from .config import WINDOW_SIZE, HORIZON, TRAIN_RATIO, VAL_RATIO, FEATURES

def chou_imputation(series, window=3):
    res = series.copy()
    invalid_idx = res[(res == 0) | (res.isna())].index
    for idx in invalid_idx:
        start = max(0, idx - window)
        end = min(len(res), idx + window + 1)
        neighbor = res[start:end]
        valid = neighbor[(neighbor > 0) & (neighbor.notna())]
        res[idx] = valid.mean() if len(valid) > 0 else series[series > 0].median()
    return res

def prepare_data(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    for col in FEATURES:
        df[col] = chou_imputation(df[col])
    return df[['Timestamp'] + FEATURES].sort_values('Timestamp').reset_index(drop=True)

def create_sequences(df, window=WINDOW_SIZE, horizon=HORIZON, features=FEATURES):
    values = df[features].values
    X, y, ts = [], [], []
    for i in range(window, len(values) - horizon):
        target_idx = i + horizon - 1
        X.append(values[i - window : i])
        y.append(values[target_idx][0])
        ts.append(df.iloc[target_idx]['Timestamp'])
    return np.array(X), np.array(y), np.array(ts)

def temporal_split(X, y, ts, train_r=TRAIN_RATIO, val_r=VAL_RATIO):
    n = len(X)
    t_idx, v_idx = int(n * train_r), int(n * (train_r + val_r))
    return {
        'X_train': X[:t_idx], 'y_train': y[:t_idx], 'ts_train': ts[:t_idx],
        'X_val': X[t_idx:v_idx], 'y_val': y[t_idx:v_idx], 'ts_val': ts[t_idx:v_idx],
        'X_test': X[v_idx:], 'y_test': y[v_idx:], 'ts_test': ts[v_idx:]
    }

def fit_scaler(split_data):
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    X_train_flat = split_data['X_train'].reshape(-1, len(FEATURES))
    X_train_scaled = scaler_X.fit_transform(X_train_flat).reshape(-1, WINDOW_SIZE, len(FEATURES))
    X_val_scaled = scaler_X.transform(split_data['X_val'].reshape(-1, len(FEATURES))).reshape(-1, WINDOW_SIZE, len(FEATURES))
    X_test_scaled = scaler_X.transform(split_data['X_test'].reshape(-1, len(FEATURES))).reshape(-1, WINDOW_SIZE, len(FEATURES))
    
    y_train_scaled = scaler_y.fit_transform(split_data['y_train'].reshape(-1, 1)).ravel()
    y_val_scaled = scaler_y.transform(split_data['y_val'].reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(split_data['y_test'].reshape(-1, 1)).ravel()
    
    return scaler_X, scaler_y, {
        'X_train': X_train_scaled, 'y_train': y_train_scaled,
        'X_val': X_val_scaled, 'y_val': y_val_scaled,
        'X_test': X_test_scaled, 'y_test': y_test_scaled,
        'ts_train': split_data['ts_train'], 'ts_val': split_data['ts_val'], 'ts_test': split_data['ts_test']
    }