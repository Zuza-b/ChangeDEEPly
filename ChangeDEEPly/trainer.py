from ChangeDEEPly.data import get_data, clean_data
from ChangeDEEPly.data import fit, transform, create_df_preprocessor, save_model
import joblib

class Trainer(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def save_model():
        filename = "KNN_shilpa_model_0.joblib"
        joblib.dump(model, filename)

if __name__ == "__main__":
    # Get and clean data
    #N = 100
    df = get_data()
    df = clean_data(df)
    X,y = create_df_preprocessor(df)
    pipe = transform(X,y)
    model = fit(X,y)
    trainer = Trainer()
    trainer.save_model(model)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # Train and save model, locally and

    #print(f"rmse: {rmse}")
    #trainer.save_model_locally()
