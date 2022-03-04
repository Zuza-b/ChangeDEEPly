import pandas as pd
import numpy as np
import datetime

def remove_duplicates(data):
    return pd.DataFrame(data).drop_duplicates()

def age_constructor(data):
    return pd.DatetimeIndex(pd.to_datetime(data['start'], format='%Y/%m/%d')).year - pd.DatetimeIndex(pd.to_datetime(data['birth'], format='%Y')).year

def array_reshape(data):
    output = np.reshape(data, (-1, 1))
    return output
