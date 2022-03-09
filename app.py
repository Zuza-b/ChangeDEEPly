import streamlit as st
import os
import time
from math import sqrt

import pandas as pd
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


from sklearn.metrics import mean_absolute_error, mean_squared_error
#st.set_page_config(layout="wide")

class App():
    # This code is different for each deployed app.
    #CURRENT_THEME = "blue"
    #IS_DARK_THEME = True
    #EXPANDER_TEXT = """
    #This is a custom theme. You can enable it by copying the following code
    #to `.streamlit/config.toml`:
    #```python
    #[theme]
    #primaryColor = "#E694FF"
    #backgroundColor = "#00172B"
    #secondaryBackgroundColor = "#0083B8"
    #textColor = "#C6CDD4"
    #font = "sans-serif"
    #```
    #"""
    def __init__(self):
        #self.Age = Age
        #self.Education = Education
        #self.category = category
        #self.Gender = Gender
        pass

    def set_page(self):
        # Set border of current theme to red, otherwise black or white
        #border_color = "red"
        #border_color = "lightgrey" if IS_DARK_THEME else "black"
        '''
        # MOOC Dropout Prediction
        '''
        #st.title('MOOC Dropout Prediction')
        MOOC_img = 'MOOC.jpg'
        #st.image(MOOC_img, width=350)

        col1, col2 = st.columns([0.7,0.3])
        with col2:
            st.image(MOOC_img, width=350)
        with col1:
            st.title('MOOC Dropout Prediction')

        content = 'Project to predict on dropout for students enrolled on MOOC(Massive Open Online Courses)'
        st.markdown(f'<p style="color:#99ff88;font-size:35px;border-radius:2%;">{content}</p>', unsafe_allow_html=True)

        #st.header('Project to predict on dropout for students enrolled on MOOC(Massive Open Online Courses)')
        #f'<p style="background-color:#0066cc;color:#77ff33;font-size:40px;border-radius:2%;">{content}</p>'
        st.subheader('Are you a Quitter or a Finisher')


        with st.expander('MODEL 1: Core Prediction'):
            st.write("""
                Features considered:

                    . Age

                    . Gender

                    . Education background

                    . Category of the course you are interested in

                Model: KNN with 5 neighbors
            """)
            st.image("KNN_img.png")

        with st.expander("MODEL 2: Behavioural Prediction"):
            st.write("""
                Features considered:

                    . Actions

                    . Time of action

                    . Other basic features like education, Age, Gender, Category

                Model: Gradient Boosting Classifier
            """)
            st.image("GradientBoost.jpg")




    def behavour_predict(self):
        #model2 = joblib.load('KNN_shilpa_model_0.joblib')
        #X = dict(

        #birth=50,
        #category='economics',
        #gender='male',
        #education='High'
        #)
        #output2 = model2.predict(X)

        #if output2 == 1:
        #    out2 = 'dropout of'
        #else:
        #    out2 = 'complete'


        st.subheader(f'This student model has to be loaded ')

    def read_csv(self):

        #adding a file uploader
        st.subheader('Do you want to upload student data behaviour over time? to check if he is going to dropout')

        file  = st.file_uploader("Upload file", type=['csv'])
        time.sleep(3)

        # get extension and read file
        if file is not None:
            #st.write('hi')
            file_details = {"FileName":file.name,"FileType":file.type,"FileSize":file.size}
            st.write(file_details)
            extension = file.name.split('.')[1]
            if extension.upper() == 'CSV':
                df = pd.read_csv(file)
                st.write(df.head())

        #predict on the student data of the instructor
        st.button('click here to predict',on_click = self.behavour_predict)
        #st.image('/home/shilpa/code/Zuza-b/ChangeDEEPly/action_counts_img.png')

    def sidebar(self):

        #side bar
        with st.sidebar.form(key='my_form'):
            st.subheader('Student Personal Info')

            Age  = st.slider('How old are you?', 0, 130, 23)

            Gender = st.selectbox('🧑 pick your gender',('male', 'female'))


            Education = st.selectbox('🏫 choose your educational background',('Doctorate', "Bachelor's", "Master's", 'High', 'Associate',
            'Middle', 'Primary'))



            category = st.selectbox(
            'which course category you are interested in',('computer', 'business', 'social science', 'art', 'literature',
            'philosophy', 'foreign language', 'math', 'engineering',
            'education', 'medicine', 'chemistry', 'history', 'electrical','economics', 'physics', 'biology'))





            submit_button = st.form_submit_button(label='Calculate!',
            on_click=self.form_callback, args=(Age, Gender, Education, category ))

            file  = st.file_uploader("Upload file", type=['csv'])
            if file is not None:
            #st.write('hi')
                file_details = {"FileName":file.name,"FileType":file.type,"FileSize":file.size}
                st.write(file_details)
                extension = file.name.split('.')[1]
                if extension.upper() == 'CSV':
                    df = pd.read_csv(file)
                    st.write(df.head())
            #return Age, Gender, Education, category

    def form_callback(self, Age, Gender, Education, category):

        st.write('please wait while we process the student info and make predictions')

        X = dict(

        birth=float(1999.0),
        category=category,
        gender=Gender,
        education=Education
        )

        #X = [['1997.0','load_video',0.0,'history','male','Doctorate']]
        X = pd.DataFrame([X])
        #st.dataframe(X)
        ## model.load
        #PATH_TO_LOCAL_MODEL = 'model.joblib'
        model = joblib.load('RF_shilpa_model_0.joblib')

        ## predict on the model
        # predict only after the user enters all the details


        output = model.predict(X)
        if output == 1:
            out = 'dropout of'
        else:
            out = 'complete'


        st.subheader(f'This student is most probable to {out} the course ')


        #st.button('upload file?')



if __name__ == '__main__':

    app = App()
    app.set_page()
    #Age, Gender, Education, category = app.sidebar()
    app.sidebar()
    #app.form_callback(Age, Gender, Education, category)
    app.read_csv()
