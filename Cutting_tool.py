import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn import svm
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Dropout

data = pd.read_csv('Final cutting tool dataset.csv')
# separeting the data and labels
x = data.drop(columns= ['Cutting tool'], axis=1)
y = data['Cutting tool']

from sklearn import preprocessing
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
y=label_encoder.fit_transform(y)

#x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.25, random_state = 0)
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.15, random_state=42)

svc_model=svm.SVC(kernel='linear',probability=True, random_state=0)
RandomForest_model=RandomForestClassifier(criterion="gini", max_depth=10, min_samples_split=2, random_state=0, n_estimators=100,max_features="sqrt") 
log_reg=LogisticRegression(random_state=1,solver='newton-cg',multi_class='ovr')
# Train a neural network model
num_classes = len(label_encoder.classes_)
input_shape = (x_train.shape[1],)

# Convert labels to one-hot encoding
y_train_nn = keras.utils.to_categorical(y_train, num_classes)
y_test_nn = keras.utils.to_categorical(y_test, num_classes)

# Define the neural network architecture
model = keras.Sequential([
    layers.Dense(50, activation='relu', input_shape=input_shape),
    layers.Dense(20, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

# Train the model
epochs = 100
batch_size = 32
history = model.fit(x_train, y_train_nn, batch_size=batch_size, epochs=epochs, validation_split=0.1,verbose=1)

log_reg=log_reg.fit(x_train,y_train)
svc_model=svc_model.fit(x_train,y_train)
RandomForest_model=RandomForest_model.fit(x_train,y_train)

pickle.dump(log_reg,open('log_model.pkl','wb'))
pickle.dump(svc_model,open('svc_model.pkl','wb'))
pickle.dump(RandomForest_model,open('RandomForest_model.pkl','wb'))
model.save('neural_network_model.h5')

log_model=pickle.load(open('log_model.pkl','rb'))
svc_model=pickle.load(open('svc_model.pkl','rb'))
RandomForest_model=pickle.load(open('RandomForest_model.pkl','rb'))
nn_model = keras.models.load_model('neural_network_model.h5')

def classify(prediction):
    if prediction==0:
        return 'Ball nose mill cutter'
    elif prediction==1:
        return 'Bore tool'
    elif prediction==2:
        return 'Chamfering end mill'
    elif prediction==3:
        return 'Corner round end mill'
    elif prediction==4:
        return 'Counter drilling tool'
    elif prediction==5:
        return 'Counter sinking tool'
    elif prediction==6:
        return 'Deep hole drill bits'
    elif prediction==7:
        return 'End mill cutter'
    elif prediction==8:
        return 'Face mill cutter'
    elif prediction==9:
        return 'Oil drill bits'
    elif prediction==10:
        return 'Reaming tool'
    elif prediction==11:
        return 'Spade drill bits'
    else:
        return 'Twist drill bits'

def app():
    #st.title("Selection of Cutting tools")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Cutting tools</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Logistic Regression','SVC','RandomForest Classifier','Neural Network']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    ft = st.selectbox('Features:', list(range(1, 11)))
    Op=st.selectbox('Operations', list(range(1,7)))
    Wm=st.selectbox('Work material', list(range(1,5)))
    Ar=st.slider('Select Aspect Ratio', 0.0, 50.0)
    Sp=st.slider('Select Speed', 0.0, 500.0)
    F=st.slider('Feed', 0.0, 1.0)
    D=st.slider('Depth of Cut', 0.0, 50.0)
    Dt=st.slider('Diameter of tool', 0.0, 400.0)
    Cm=st.selectbox('Cutting tool material', list(range(1,5)))
    
    
    inputs=[[ft,Op,Wm,Ar,Sp,F,D,Dt,Cm]]
    
    if st.button('Classify'):
        if option=='Logistic Regression':
            st.success(classify(log_model.predict(inputs)))
        elif option=='RandomForest Classifier':
            st.success(classify(RandomForest_model.predict(inputs)))
        elif option == 'Neural Network':
            prediction = model.predict(np.array(inputs))
            predicted_class_index = np.argmax(prediction[0])
            predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
            st.success(predicted_class)
        else:
            st.success(classify(svc_model.predict(inputs)))
            
    # List of model names
    model_names = ['Logistic Regression', 'SVC', 'RandomForest Classifier', 'Neural Network']

    # List of trained models
    trained_models = [log_reg,svc_model, RandomForest_model, nn_model]

    if __name__=='__app__':
        app()