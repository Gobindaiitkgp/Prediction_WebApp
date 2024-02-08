import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Final machine tool.csv')
# separeting the data and labels
x = data.drop(columns= ['Machine'], axis=1)
y = data['Machine']

from sklearn import preprocessing
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
y=label_encoder.fit_transform(y)

#x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.25, random_state = 0)
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.15, random_state=0)

svc_model=SVC(kernel = 'linear', random_state = 1)
RandomForest_model=RandomForestClassifier(criterion="gini",max_depth=8,min_samples_split=2,random_state=5) 

svc_model=svc_model.fit(x_train,y_train)
RandomForest_model=RandomForest_model.fit(x_train,y_train)

pickle.dump(svc_model,open('svc_model.pkl','wb'))
pickle.dump(RandomForest_model,open('RandomForest_model.pkl','wb'))

svc_model=pickle.load(open('svc_model.pkl','rb'))
RandomForest_model=pickle.load(open('RandomForest_model.pkl','rb'))

def classify(prediction):
    if prediction==0:
        return 'Boring Machine'
    elif prediction==1:
        return 'CNC horizontal milling machine'
    elif prediction==2:
        return 'CNC vertical milling machine'
    elif prediction==3:
        return 'Drilling machine'
    elif prediction==4:
        return 'Horizontal milling machine'
    elif prediction==5:
        return 'Machining centre'
    else:
        return 'Vertical milling machine'

def app():
    #st.title("Selection of Machine tools")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Machine tools</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['SVC','RandomForest Classifier']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    ft = st.selectbox('Features:', list(range(1, 11)))
    Op=st.selectbox('Operations', list(range(1,7)))
    x=st.slider('Select X(mm)', 0.0, 1500.0)
    y=st.slider('Select Y(mm)', 0.0, 1500.0)
    z=st.slider('Select Z(mm)', 0.0, 1500.0)
    Di=st.slider('Select Max diameter(mm)', 0.0, 200.0)
    SS=st.slider('Select Spindle speed(rpm)', 0.0, 12000.0)
    Fx=st.slider('Select Longitudinal feed(mm/rev)', 0.0, 2.0)
    Fy=st.slider('Select Cross feed(mm/rev)', 0.0, 2.0)
    Fz=st.slider('Select Vertical feed(mm/rev)', 0.0, 2.0)
    P=st.slider('Select Power(KW)', 0.0, 20.0)
    L=st.slider('Select Load(Kg)', 0.0, 2000.0)
    inputs=[[ft,Op,x,y,z,Di,SS,Fx,Fy,Fz,P,L]]
    
    if st.button('Classify'):
        if option=='RandomForest Classifier':
            st.success(classify(RandomForest_model.predict(inputs)))
        else:
            st.success(classify(svc_model.predict(inputs)))
            
    # List of model names
    model_names = ['SVC', 'RandomForest Classifier']

    # List of trained models
    trained_models = [svc_model, RandomForest_model]

    if __name__=='__app__':
        app()