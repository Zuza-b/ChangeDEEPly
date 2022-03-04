import streamlit as st
import os
from math import sqrt

import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

'''
# ChangeDEEPly
'''
#st.title('Change DEEPly')

st.header('Predicting whether You are a Quitter or Finisher')
st.markdown('''

 Basic predicting:
- Features considered
    . Age
    . Gender
    . Education background
    . Course enrolled

- Some more features for predicting:
    . behaviour of user over time considering his various actions
''')


## take the values from the user



st.markdown(' We need your personal info to predict whether you are a drop out or not')


BirthYear = st.date_input(' 📅 pick your birth year')

Age  = st.slider('How old are you?', 0, 130, 25)
#st.write("I'm ", Age, 'years old')

Gender = st.selectbox('🧑 pick your gender',('Male', 'Female'))
st.write('You selected:', Gender)

Education = st.selectbox('🏫 choose your educational background',('Doctorate', "Bachelor's", "Master's", 'High', 'Associate',
'Middle', 'Primary'))
st.write('You selected:', Education)


category = st.selectbox(
'which course category you are interested in',('computer', 'business', 'social science', 'art', 'literature',
'philosophy', 'foreign language', 'math', 'engineering',
'education', 'medicine', 'chemistry', 'history', 'electrical','economics', 'physics', 'biology'))
st.write('You selected:', category)


st.write('please wait while we process your data and make predictions')

#X = dict(

#birth=BirthYear,
#education=float(Education),
#gender=int(Gender),
#category=category)

#
X = dict(

birth=float(1997.0),
action = 'load_video',
timediff=float(0.0),
category='hostory',
gender='male',
education='Doctorate')

#X = [['1997.0','load_video',0.0,'history','male','Doctorate']]
X = pd.DataFrame([X])
st.write(X)
## model.load
#PATH_TO_LOCAL_MODEL = 'model.joblib'
model = joblib.load('KNN_BaseModel.joblib')




## predict on the model

output = model.predict(X)
st.write(f'there is a {output} probability of you completing the course')
