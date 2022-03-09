### ALL IMPORTS ----- LATER ADDING AT THE BEGINNING ###

# MACHINE LEARNING
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
#from sklearn import set_config; set_config(display='diagram')
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_union
from sklearn.compose import make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

class Encoder():
    def __init__(self):
        pass

    def fit(self, X, y):
        #self.pipe.fit_transform(self.X, self.y)
        knn_model = self.pipe.fit(X, y)
        return knn_model

    def transform(self, X, y):
        # Impute then Scale for numerical variables:
        num_transformer = Pipeline([
        ('scaler', StandardScaler())])

        # Encode categorical variables
        cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)

        # Paralellize "num_transformer" and "One hot encoder"
        preprocessor = ColumnTransformer([
        ('num_tr', num_transformer, ['birth']),
        ('cat_tr', cat_transformer, ['education', 'gender', 'category'])],
        remainder='drop')
        #X_transformed = preprocessor.transform(X)

        #print(X_train.head(3))
        #print(pd.DataFrame(X_train_transformed).head(3))
        self.pipe = make_pipeline(preprocessor, RandomForestClassifier())

        return self.pipe

    def create_df_preprocessor(self,df):
        X = df[['birth', 'category', 'gender', 'education']]
        X = X.iloc[:2000]
        y = df['truth']
        y = y.iloc[:2000]

        return X,y
