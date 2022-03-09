from ChangeDEEPly.data.data import get_data
from ChangeDEEPly.encoders_2 import Encoder2 #fit, transform, create_df_preprocessor
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

class Trainer2(object):
    def __init__(self):
        pass

    def save_model(self,model):
        filename = "../TOSOLVE_PROBLEM_GBC_model2.joblib"
        joblib.dump(model, filename)

if __name__ == "__main__":
    # Get and clean data
    #N = 100
    df = get_data()

    #df = clean_data(df)
    encoder = Encoder2()
    X, y = encoder.preprocess_pipeline(df)

    #df = encoder.fit_preprocessing_pipeline(df)
    #print(df.head(2))
    #X,y = encoder.create_df_preprocessor2(df)

    #pipe = encoder.transform(X,y)
    #model = encoder.fit(X,y)
    # instanciate the model
    model = GradientBoostingClassifier()
    # training the model
    model.fit(X,y)
    #print(model)
    trainer = Trainer2()
    trainer.save_model(model) ######### LATER UNCOMMENT
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Train and save model, locally and

    #print(f"rmse: {rmse}")
    #trainer.save_model_locally()
    #X = dict(

    #birth=float(1997.0),
    #category='history',
    #gender='male',
    #education='Doctorate')

    df_test = get_data()
    encoder_test = Encoder2()
    X_test, y_test = encoder_test.preprocess_pipeline(df_test)
    #X_test = pd.DataFrame(X_test.iloc[44,])
    #print(X_test)
    prediction = model.predict(X_test.iloc[[4],:])

#
    ##X = [['1997.0','load_video',0.0,'history','male','Doctorate']]
    #X = pd.DataFrame([X])
    #output = model.predict(X)
    #print(output)
    #print(X_test.iloc[[4],:])
    print('DONE')
