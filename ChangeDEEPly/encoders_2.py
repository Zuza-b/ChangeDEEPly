# MACHINE LEARNING
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
#from sklearn import set_config; set_config(display='diagram')
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
# Zuzanna stuff
from ChangeDEEPly.feature_encoding import *

class Encoder2 (): ######## added by David
    def __init__(self):
        pass

    def preprocess_pipeline(self, df):
        # copying the stuff from the notebook

        gender_pipe = Pipeline([
        ('ohe_gender', OneHotEncoder(drop='if_binary', sparse = False, handle_unknown='ignore'))
        ])

        category_edu_pipe = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse = False))
        ])

        keep_unchanged_pipe = Pipeline([
        ('keep_unchanged', FunctionTransformer(keep_unchanged))#,
        #('reshape', FunctionTransformer(array_reshape))
        ])

        age_pipe = Pipeline([
        ('age_calc', FunctionTransformer(age_constructor)),
        ('reshape', FunctionTransformer(array_reshape)),
        ('stdscaler', StandardScaler())
        ])

        actions_pipe = Pipeline([
        ('ohe_action', FunctionTransformer(action_time)),
        ('combining_actions', FunctionTransformer(combining_actions)),
        ('stdscaler', StandardScaler())
        ])

        basic_encoding_pipe = ColumnTransformer([
        ('keep_unchanged', keep_unchanged_pipe, ["truth","username", "id"]),
        ('category_edu_pipe', category_edu_pipe, ["category", "education"]),
        ('gender_pipe', gender_pipe, ["gender"])
        ], remainder="drop")

        preprocessing_pipe = FeatureUnion([
        ('basic_encoding_pipe', basic_encoding_pipe),
        ('age', age_pipe)
        ])

        removing_duplicates_pipe = Pipeline([
        ('preprocessing_pipe', preprocessing_pipe),
        ('remove_duplicates', FunctionTransformer(remove_duplicates))
        ])

        merge_pipe = FeatureUnion([
        ('all_without_duplicates', removing_duplicates_pipe),
        ('actions', actions_pipe)
        ])

        df_trans = pd.DataFrame(merge_pipe.fit_transform(df))
        X = df_trans.drop(columns=[0])
        X = X.set_index([1,2])
        y = df_trans[0]

        return X, y

    #def fit_preprocessing_pipeline(self, df):
    #    preprocess_pipeline = self.preprocess_pipeline(df)
    #    #preprocess_pipeline.fit(df)
    #    df_trans = pd.DataFrame(preprocess_pipeline.fit_transform(df))
    #    return df_trans
#
    #def create_df_preprocessor2(self, df):
    #    df_trans = self.fit_preprocessing_pipeline(df)
    #    X = df_trans.drop(columns=[0])
    #    X = X.set_index([1,2])
    #    y = df_trans[0]
#
    #    return X,y
#
