from ChangeDEEPly.data.data import get_data, clean_data
from ChangeDEEPly.encoders import Encoder #fit, transform, create_df_preprocessor
import joblib
import pandas as pd




class Trainer(object):
    def __init__(self):
        pass

    def save_model(self,model):
        filename = "../RF_Base_model_0.joblib"
        joblib.dump(model, filename)

if __name__ == "__main__":
    # Get and clean data
    #N = 100
    df = get_data()
    #df = clean_data(df)
    encoder = Encoder()
    X,y = encoder.create_df_preprocessor(df)
    pipe = encoder.transform(X,y)
    model = encoder.fit(X,y)


    #print(model)
    trainer = Trainer()
    trainer.save_model(model)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Train and save model, locally and

    #print(f"rmse: {rmse}")
    #trainer.save_model_locally()
    X = dict(

    birth=float(1997.0),
    category='history',
    gender='male',
    education='Doctorate')
#
    #X = [['1997.0','load_video',0.0,'history','male','Doctorate']]
    X = pd.DataFrame([X])
    output = model.predict(X)
    print(output)
