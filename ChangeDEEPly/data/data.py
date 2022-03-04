import pandas as pd
import numpy as np



def get_data(nrows=1000000):
    """get data """
    df = pd.read_csv('../raw_data/df_final.csv',nrows)
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
