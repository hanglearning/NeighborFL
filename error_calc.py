""" Error calculation functions """

# Prepare error calculation functions

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

def get_MAE(y_true, y_pred):
    try:
        return round(mean_absolute_error(y_true, y_pred), 2)
    except ValueError:
        return np.nan
  

def get_MSE(y_true, y_pred):  
    try:
        return round(mean_squared_error(y_true, y_pred, squared=True), 2)
    except ValueError:
        return np.nan

def get_RMSE(y_true, y_pred): 
    try:
        return round(mean_squared_error(y_true, y_pred, squared=False), 2)
    except ValueError:
        return np.nan

def get_MAPE(y_true, y_pred):  
    try:
        return round(mean_absolute_percentage_error(y_true, y_pred), 2)
    except ValueError:
        return np.nan