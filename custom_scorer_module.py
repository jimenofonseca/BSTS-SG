from sklearn.metrics import mean_squared_error
import math
import numpy as np

def scorer_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    rsme = math.sqrt(mse)
    return rsme

def scorer_quantile(y_true, y_pred, quantile=0.5):
    e = y_true - y_pred
    return np.mean(np.maximum(quantile * e, (quantile - 1) * e))