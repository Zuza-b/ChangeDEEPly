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


from streamlit_option_menu import option_menu

from sklearn.metrics import mean_absolute_error, mean_squared_error
#st.set_page_config(layout="wide")


class App():
    # This code is different for each deployed app.

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
    st.set_page_config(
                page_title='Streamlit Demonstration App',
                # set Streamlit logotype as the favicon
                #page_icon="Streamlit_Logo_1.jpg", # None, ":memo:...
                #layout='wide', # centered, wide
                #initial_sidebar_state='expanded' # auto, expanded, collapsed
            )


    def __init__(self):
        #self.Age = Age
        #self.Education = Education
        #self.category = category
        #self.Gender = Gender
        pass


    def set_page(self):
        # Set border of current theme to red, otherwise black or white
        #CURRENT_THEME = "blue"
        #IS_DARK_THEME = True
        #border_color = "red"
        #border_color = "lightgrey" if IS_DARK_THEME else "black"
        '''
        # MOOC Dropout Prediction
        '''
        #st.title('MOOC Dropout Prediction')
        MOOC_img = 'MOOC.jpg'
        #st.image(MOOC_img, width=350)
        with st.container():
            col1, col2 = st.columns([0.7,0.3])
            with col2:
                st.image(MOOC_img, width=350)
            with col1:
                st.title('MOOC Dropout Prediction')

        content = 'Project to predict on dropout for students enrolled on MOOC(Massive Open Online Courses)'
        st.markdown(f'<p style="color:#000080;font-size:35px;border-radius:5%;">{content}</p>', unsafe_allow_html=True)

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
        output2 = [1,0,1,0]
        list(output2)
        user_col = ['user_1','user_2','user_3', 'user_4']
        list1 = []
        for i in output2:
            if i == int(1):
                list1.append('dropout')
            else:
                list1.append('complete')
        data_tuples = list(zip(user_col,list1))
        o = pd.DataFrame(data_tuples, columns=['Username','Most Probable to'])
        st.write(o)



        #st.subheader(f'This student model has to be loaded ')

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

    def student_info(self):


        with st.form(key='my_form'):
            st.subheader('Student Personal Info')

            #Age  = st.slider('How old are you?', 0, 130, 23)
            Age = st.number_input("birth year")
            Gender = st.selectbox('🧑 pick your gender',('male', 'female'))


            Education = st.selectbox('🏫 choose your educational background',('Doctorate', "Bachelor's", "Master's", 'High', 'Associate',
            'Middle', 'Primary'))



            category = st.selectbox(
            'which course category you are interested in',('computer', 'business', 'social science', 'art', 'literature',
            'philosophy', 'foreign language', 'math', 'engineering',
            'education', 'medicine', 'chemistry', 'history', 'electrical','economics', 'physics', 'biology'))





            submit_button = st.form_submit_button(label='Calculate!',
            on_click=self.form_callback, args=(Age, Gender, Education, category ))

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

        st.subheader(f'This student is most probable to {out} the course')
        #st.write('here is your output')

        #st.button('upload file?')



if __name__ == '__main__':

    app = App()
    #app.set_page()
    #Age, Gender, Education, category = app.sidebar()
    #app.student_info()
    #app.form_callback(Age, Gender, Education, category)
    #app.read_csv()

    with st.sidebar:
        selected = option_menu("MOOC Menu", ["Models", 'Student_PI', 'Instructor'])
        st.sidebar.title("About")
        st.sidebar.write(
        """
        This is an app to predict if a student is a dropout or not from an online training.
        Created with passion for learning. Created for you to succeed, to learn and grow.
        Created by a team of dedicated and skilful data scientists.
        """
    )
    if selected == "Models":
        app.set_page()
    if selected == 'Student_PI':
        app.student_info()
    if selected == 'Instructor':
        app.read_csv()
