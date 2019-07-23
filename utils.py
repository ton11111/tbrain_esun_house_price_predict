import numpy as np

def hit_rate(y_true, y_pred):
    return np.mean(((np.abs((y_pred - y_true) / y_true)) <= 0.1).astype(float))

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true))

def score_this_game(y_true, y_pred):
    hr = hit_rate(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return round(hr, 4) * 10000 + (1 - min(1, mape))

def calc_mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

