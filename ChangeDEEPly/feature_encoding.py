import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def remove_duplicates(data):
    return pd.DataFrame(data).drop_duplicates()

def age_constructor(data):
    return pd.DatetimeIndex(pd.to_datetime(data['start'], format='%Y/%m/%d')).year - pd.DatetimeIndex(pd.to_datetime(data['birth'], format='%Y')).year

def array_reshape(data):
    output = np.reshape(data, (-1, 1))
    return output

def action_time(data):
    ohe_action = OneHotEncoder(sparse = False) # Instanciate encoder
    ohe_action.fit(data[['action']]) # Fit encoder
    action_encoded = ohe_action.transform(data[['action']])
    action_X = pd.DataFrame(action_encoded)
    timediff = list(data['timediff'])
    for row in range(len(action_X.index)):
        action_X.iloc[row,action_X.columns[action_X.iloc[row,] == 1][0]] = timediff[row]
    action_X['username'] = data[['username']]
    action_X['id'] = data[['id']]
    return action_X

def combining_actions(dataframe):
    output_df = dataframe.groupby(['username', 'id']).sum()
    return output_df

def keep_unchanged(data):
    return pd.DataFrame(data[["truth", "username", "id"]])

def calc_percentage_course(df):
    df['course_start'] = pd.to_datetime(df['start'])
    df['course_start'] = df.course_start.values.astype(np.int64) // 10 ** 9
    df['course_end'] = pd.to_datetime(df['end'])
    df['course_end'] = df.course_end.values.astype(np.int64) // 10 ** 9
    #df = df.drop(['start'], axis = 1)
    #df = df.drop(['end'], axis = 1)
    df['percent_course'] = (df['timestamp'] - df['course_start'])/(df['course_end'] - df['course_start'])
    users_min_30per_comp = df[df['percent_course'] >= 0.3]['username'].unique()
    df = df[df['username'].isin(users_min_30per_comp)][df['percent_course'] < 0.3]
    return df
