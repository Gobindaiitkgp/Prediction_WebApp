import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
data = pd.read_csv('Cutting parameters.csv')

# Separate features and target variables
X = data.drop(columns=['Depth of cut','Feed','Spindle speed'], axis=1)
y = data[['Depth of cut','Feed','Spindle speed']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

# Train the Random Forest model
RandomForest_model = RandomForestRegressor(n_estimators=100, random_state=0)
RandomForest_model.fit(X_train, y_train)

# Load the trained model
# RandomForest_model = pickle.load(open('RandomForest_model.pkl', 'rb'))

# Prediction function
def predict_parameters(model, inputs):
    return model.predict(inputs)

# Streamlit application
def app():
    #st.title("Selection of Machining parameters")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Machining parameter</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['RandomForest Regressor']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    ft = st.selectbox('Features:', list(range(1, 11)))
    Op=st.selectbox('Operations', list(range(1,6)))
    Wm=st.selectbox('Work material', list(range(1,5)))
    Cm=st.selectbox('Cutting tool material', list(range(1,5)))
    Sp=st.slider('Select Speed', 0.0, 500.0)
    F=st.slider('Feed per tooth', 0.0, 1.0)
    T=st.slider('No. of tooth',0.0,10.0)
    Dt=st.slider('Diameter of tool', 0.0, 400.0)
    
    
    inputs=[[ft,Op,Wm,Cm,Sp,F,T,Dt]]
    
    if st.button('Predictions'):
        if option=='RandomForest Regressor':
            predictions = predict_parameters(RandomForest_model, inputs)
            predictions_formatted = [[f'{val:.2f}' for val in sublist] for sublist in predictions]
        st.success(predictions_formatted)
        

if __name__ == '__main__':
    app()
