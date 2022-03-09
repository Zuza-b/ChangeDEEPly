import streamlit as st
import os
import time
from math import sqrt

import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
#st.set_page_config(layout="wide")

# This code is different for each deployed app.
#CURRENT_THEME = "blue"
#IS_DARK_THEME = True
#EXPANDER_TEXT = """
#    This is a custom theme. You can enable it by copying the following code
#    to `.streamlit/config.toml`:
#    ```python
#    [theme]
#    primaryColor = "#E694FF"
#    backgroundColor = "#00172B"
#    secondaryBackgroundColor = "#0083B8"
#    textColor = "#C6CDD4"
#    font = "sans-serif"
#    ```
#    """

# Set border of current theme to red, otherwise black or white
#border_color = "red"
#border_color = "lightgrey" if IS_DARK_THEME else "black"
'''
MOOC Dropout Prediction
'''
MOOC_img = '/home/shilpa/code/Zuza-b/ChangeDEEPly/MOOC.jpg'
st.image(MOOC_img, width=350)



st.title('Project to predict on dropout for students enrolled on MOOC(Massive Open Online Courses)')

st.header('Are you a Quitter or a Finisher')
st.markdown('''

 Basic predicting:
- Features considered
    . Age
    . Gender
    . Education background
    . Category of the course you are interested in

- Some more features for predicting:
    . behaviour of user over time considering his various actions. We get this info from the instructor
    uploaded csv file at some point during the course to predict whether the student drop outs or not
''')

def read_csv(file):

    pass


## take the values from the user
def form_callback():

    X = dict(

    birth=float(1999.0),
    category=category,
    gender=Gender,
    education=Education
    )

#
#X = dict(

#birth=float(1999.0),
#category='computer',
#gender='female',
#education='High')

#X = [['1997.0','load_video',0.0,'history','male','Doctorate']]
    X = pd.DataFrame([X])
    st.dataframe(X)
## model.load
#PATH_TO_LOCAL_MODEL = 'model.joblib'
    model = joblib.load('KNN_David_model_0.joblib')

## predict on the model
# predict only after the user enters all the details


    output = model.predict(X)
    if output == 1:
        out = 'dropout of'
    else:
        out = 'complete'
    st.write(f'You are most probable to {out} the course')

    st.write('Do you want to upload student data behaviour over time?')


    #adding a file uploader

    uploaded_file  = st.file_uploader("Upload file", type=['csv'])
    time.sleep(10)
    st.write(f'file uploaded is {uploaded_file}')
    st.form_submit_button(label='upload!', on_click=read_csv(uploaded_file))
    if uploaded_file  is not None:
        st.form_submit_button(label='upload!', on_click=read_csv(uploaded_file))
        st.write('im here inside the loop if')
        df_sample = pd.read_csv(uploaded_file )
        time.sleep(5)
        df_sample = df_sample.sample(10)
        st.dataframe(df_sample)
        st.dataframe(data=[[1,2,3]],columns=['a','b','c'])
    st.form_submit_button(label='upload!', on_click=read_csv(uploaded_file))
    time.sleep(3)


#side bar
with st.sidebar.form(key='my_form'):
    st.subheader('Student Personal Info')

    Age  = st.slider('How old are you?', 0, 130, 25)

    Gender = st.selectbox('🧑 pick your gender',('male', 'female'))


    Education = st.selectbox('🏫 choose your educational background',('Doctorate', "Bachelor's", "Master's", 'High', 'Associate',
    'Middle', 'Primary'))



    category = st.selectbox(
    'which course category you are interested in',('computer', 'business', 'social science', 'art', 'literature',
    'philosophy', 'foreign language', 'math', 'engineering',
    'education', 'medicine', 'chemistry', 'history', 'electrical','economics', 'physics', 'biology'))



    st.write('please wait while we process your data and make predictions')

    submit_button = st.form_submit_button(label='Calculate!', on_click=form_callback)
