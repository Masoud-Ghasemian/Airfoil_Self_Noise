 
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import xgboost
from PIL import Image


# loading the trained model
#model_path = "/content/RFModel.pkl"
scaler_path = "/content/scaler.pkl"
#model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

def model_selection(model):
  import pickle
  if model == "Linear Regression":
    model_path = "/content/LRModel.pkl"
  elif model == "K Nearest Neighbor":
    model_path = "/content/KNNModel.pkl"
  elif model == "Random Forest":
    model_path = "/content/RFModel.pkl"
  else:
    model_path = "/content/XGBModel.pkl"
  
  model = pickle.load(open(model_path, 'rb'))

  return model

def prediction(Frequency, AoA, Chord, Velocity, Displacement, model):
  Chord /= 100
  Displacement /= 1000
  columns = ['Frequency', 'AoA', 'Chord',  'Velocity', 'Displacement']
  data = {'Frequency' : [Frequency], 'AoA' : [AoA], 'Chord' : [Chord],  'Velocity' : [Velocity], 'Displacement': [Displacement]}
  df = pd.DataFrame(data = data)
  df[columns]=df[columns].astype(float)
  df[columns] = scaler.transform(df[columns])



  
  # Making predictions
  model = model_selection(model)
  prediction = model.predict(df.values)
     
  return prediction[0]
         
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:grey;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Airfoil Self-noise Prediction ML App</h1> 
    </div> 
    """
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    #age = st.number_input("Age")
    Frequency = st.slider('Frequency (Hz)', 200, 20000, 2000)
    AoA = st.slider('Angle of Attack (Degree)', 0, 20, 10)
    Chord = st.slider('Chord Length (cm)', 2, 20, 10)
    Velocity = st.slider('Freestream Velocity (m/s)', 30, 70, 50)
    Displacement = st.slider('Displacement (mm)', 1, 60, 25)
    model = st.selectbox('Prediction Model',("...","Linear Regression", "K Nearest Neighbor", "Random Forest","XGBoost"))

    result =""

    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(Frequency, AoA, Chord, Velocity, Displacement, model)
        st.success('Sound Pressure Level (SPL) : %5.2f dB'%(result))
     
if __name__=='__main__': 
    main()