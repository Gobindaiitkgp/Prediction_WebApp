
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import pickle
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from sklearn import svm
from tensorflow import keras
from tensorflow.keras import layers

data = pd.read_csv('MachiningOperation (1).csv')
# separeting the data and labels
x = data.drop(columns= ['MachiningOperation'], axis=1)
y = data['MachiningOperation']

from sklearn import preprocessing
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
y=label_encoder.fit_transform(y)

#x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.25, random_state = 0)
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.15, random_state=42)


log_reg=LogisticRegression(C=100,random_state=1,solver='lbfgs',multi_class='ovr')
svc_model=svm.SVC(kernel='linear',probability=True, random_state=0)
RandomForest_model=RandomForestClassifier(criterion="gini",max_depth=10,min_samples_split=2,random_state=0,n_estimators=100,max_features="sqrt") 
DecisionTree_model=DecisionTreeClassifier(criterion='gini',min_samples_split=2)
gaussianNB_model = GaussianNB()
knn_model=KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)

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
batch_size = 64
history = model.fit(x_train, y_train_nn, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test_nn))


log_reg=log_reg.fit(x_train,y_train)
svc_model=svc_model.fit(x_train,y_train)
RandomForest_model=RandomForest_model.fit(x_train,y_train)
DecisionTree_model=DecisionTree_model.fit(x_train,y_train)
gaussianNB_model=gaussianNB_model.fit(x_train,y_train)
knn_model=knn_model.fit(x_train,y_train)


pickle.dump(log_reg,open('log_model.pkl','wb'))
pickle.dump(svc_model,open('svc_model.pkl','wb'))
pickle.dump(RandomForest_model,open('RandomForest_model.pkl','wb'))
pickle.dump(DecisionTree_model,open('DecisionTree_model.pkl','wb'))
pickle.dump(gaussianNB_model,open('gaussianNB_model.pkl','wb'))
pickle.dump(knn_model,open('knn_model.pkl','wb'))
model.save('neural_network_model.h5')


log_model=pickle.load(open('log_model.pkl','rb'))
svc_model=pickle.load(open('svc_model.pkl','rb'))
RandomForest_model=pickle.load(open('RandomForest_model.pkl','rb'))
DecisionTree_model=pickle.load(open('DecisionTree_model.pkl','rb'))
gaussianNB_model=pickle.load(open('gaussianNB_model.pkl','rb'))
knn_model=pickle.load(open('knn_model.pkl','rb'))
nn_model = keras.models.load_model('neural_network_model.h5')

def classify(prediction):
    if prediction==0:
        return 'Drilling'
    elif prediction==1:
        return 'Drilling-CounterBoring'
    elif prediction==2:
        return 'Drilling-CounterBoring-FinishBroaching'
    elif prediction==3:
        return 'Drilling-CounterBoring_RoughReaming_SemifinishReaming'
    elif prediction==4:
        return 'Drilling-CounterBoring_RoughReaming_SemifinishReaming'
    elif prediction==5:
        return 'Drilling-RoughBoring-SemifinishBoring'
    elif prediction==6:
        return 'Drilling-RoughBoring-SemifinishBoring-DiamondBoring'
    elif prediction==7:
        return 'Drilling-RoughBoring-SemifinishBoring-FinishBoring'
    elif prediction==8:
        return 'Drilling-RoughBoring-SemifinishBoring-Grinding-Honing'
    elif prediction==9:
        return 'Drilling-RoughBoring-SemifinishBoring-Grinding-Lapping'
    elif prediction==10:
        return 'Drilling-RoughBoring-SemifinishBoring-RoughGrinding-FinishGrinding'
    elif prediction==11:
        return 'Drilling-RoughBoring-SemifinishBoring-RoughGrinding-SemifinishGrinding'
    elif prediction==12:
        return 'RoughMilling'
    elif prediction==13:
        return 'RoughMilling-SemifinishMilling'
    elif prediction==14:
        return 'RoughMilling-SemifinishMilling-FinishMilling'
    elif prediction==15:
        return 'RoughMilling-SemifinishMilling-Grinding-Lapping'
    elif prediction==16:
        return 'RoughMilling-SemifinishMilling-RoughGrinding'
    else:
        return 'RoughMilling-SemifinishMilling-Grinding-Superfinishing'
    
def save_predictions_to_excel(predictions):
    df = pd.DataFrame({
        'Predicted Result': predictions
    })

    file_path = 'E:\PhD work\Process plan\Predicted result.xlsx'
    df.to_excel(file_path, index=False)
    return file_path

def app():
    #st.title("Selection of Machining Operations")
    html_temp = """
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Machining Operations</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    activities=['Logistic Regression','SVC','RandomForest Classifier','DecisionTree Classifier','GaussianNB','Kneighbors Classifier','Neural Network']
    option=st.sidebar.selectbox('Which model would you like to use?',activities)
    st.subheader(option)
    
    #ft=st.slider('Features',1.0,10.0)
    #ft = st.text_input('Features',1.0,10.0 )
    ft = st.selectbox('Features:', list(range(1, 11)))
    
    try:
        ft = float(ft)
        if ft < 1.0 or ft > 11.0:
            st.warning('Please enter a value between 1.0 and 10.0')
    except ValueError:
        st.warning('Please enter a valid numeric value')
    D=st.slider('Select Diameter(mm)', 0.0, 250.0)
    Dp=st.slider('Select Depth(mm)', 0.0, 250.0)
    L=st.slider('Select Length(mm)', 0.0, 250.0)
    W=st.slider('Select Width(mm)', 0.0, 250.0)
    R=st.slider('Select Radius(mm)', 0.0, 30.0)
    A=st.slider('Select Angle(0)', 0.0, 90.0)
    Dis=st.slider('Select Distance(mm)', 0.0, 18.0)
    tl=st.slider('Select Tolerance(µm)', 0.0, 720.0)
    sf=st.slider('Select Surface Finish(µm)', 0.0, 80.0)
    inputs=[[ft,D,Dp,L,W,R,A,Dis,tl,sf]]
    
    if st.button('Classify'):
        if option=='Logistic Regression':
            st.success(classify(log_model.predict(inputs)))
        elif option=='RandomForest Classifier':
            st.success(classify(RandomForest_model.predict(inputs)))
        elif option=='DecisionTree Classifier':
            st.success(classify(DecisionTree_model.predict(inputs)))
        elif option=='GaussianNB':
            st.success(classify(gaussianNB_model.predict(inputs)))
        elif option=='Kneighbors Classifier':
            st.success(classify(knn_model.predict(inputs)))
        elif option == 'Neural Network':
            prediction = model.predict(np.array(inputs))
            predicted_class_index = np.argmax(prediction[0])
            predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
            st.success(predicted_class)
        else:
            st.success(classify(svc_model.predict(inputs)))
            
    # Save predicted results to Excel
        if st.button('Save Predictions to Excel'):
            file_path = save_predictions_to_excel(prediction)
            st.success(f'Predictions saved to {file_path}')

# List of model names
model_names = ['Logistic Regression', 'SVC', 'RandomForest Classifier',
               'DecisionTree Classifier', 'GaussianNB', 'Kneighbors Classifier','Neural Network']

# List of trained models
trained_models = [log_reg, svc_model, RandomForest_model, DecisionTree_model,
                  gaussianNB_model, knn_model,nn_model]

if __name__=='__app__':
    app()