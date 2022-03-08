import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer

def remove_duplicates(dataframe):
    return pd.DataFrame(dataframe).drop_duplicates()

age_constructor = FunctionTransformer(lambda data: pd.DatetimeIndex(pd.to_datetime(data['start'], format='%Y-%m-%d')).year - pd.DatetimeIndex(pd.to_datetime(data['birth'], format='%Y')).year)
array_reshape = FunctionTransformer(lambda data: np.reshape(data, (-1, 1)))
rp = FunctionTransformer(lambda data: remove_duplicates(data))
