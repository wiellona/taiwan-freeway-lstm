import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def evaluate_model(model, y_true, y_pred_scaled, scaler_y):
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true_orig = scaler_y.inverse_transform(y_true.reshape(-1, 1)).ravel() if len(y_true.shape)==1 else y_true
    return {
        'mae': mean_absolute_error(y_true_orig, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true_orig, y_pred)),
        'y_true': y_true_orig, 'y_pred': y_pred
    }

def evaluate_peak_nonpeak(y_true, y_pred, timestamps):
    ts = pd.to_datetime(timestamps)

    is_weekday = ts.dayofweek < 5
    is_peak_hour = ts.hour.isin([7, 8, 17, 18])
    mask = is_weekday & is_peak_hour
    def calc(m):
        if m.sum() == 0: return {'mae': 0, 'rmse': 0}
        return {'mae': mean_absolute_error(y_true[m], y_pred[m]),
                'rmse': np.sqrt(mean_squared_error(y_true[m], y_pred[m]))}
    return {'peak': calc(mask), 'non_peak': calc(~mask)}