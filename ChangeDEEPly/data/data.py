import pandas as pd
import numpy as np
import os


def get_data():
    """get data """
    df = pd.read_csv('/home/shilpa/code/Zuza-b/ChangeDEEPly/raw_data/df_full_withtime.csv',nrows = 5000)

    return df


def clean_data(df):
    unused_column = "Unnamed: 0"
    if unused_column in df.keys():
        df = df.drop(axis=1, columns=["Unnamed: 0"])
    df = df.dropna(how='any', axis='rows')
    df['Datetime'] = pd.to_datetime(df['time'])
    df = df.drop(['time'], axis = 1)
    ## try to convert datetime to timestamp
    df['timestamp'] = df.Datetime.values.astype(np.int64) // 10 ** 9
    df = df.drop(['Datetime'], axis = 1)
    df = df.sort_values(['username','timestamp'])
    df['timediff'] = pd.DataFrame(df.groupby('username').timestamp.diff().fillna(0))

    return df


if __name__ == '__main__':
    df = get_data()
    #df = clean_data(df)
    print(df.head())
