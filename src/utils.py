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


class ReverseEngineer:
    def __init__(self, target):
        self._target = target

    def transform(self, x):
        if self._target == 'total_price':
            return ((x - 196452) / 200) ** (2 / 3) * 10000
        elif self._target == 'building_area':
            return x ** (1 / 1.31) / 11.61593
        else:
            raise NotImplementedError()

    def reverse_transform(self, x):
        if self._target == 'total_price':
            return (x / 10000.0) ** (3.0 / 2.0) * 200 + 196452
        elif self._target == 'building_area':
            return (x * 11.61593) ** 1.31
        else:
            raise NotImplementedError()
