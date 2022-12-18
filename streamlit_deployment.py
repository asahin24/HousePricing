import streamlit as st

header = st.container()
dataset = st.container()
predicitons = st.container()

with header:
    st.title('Linear Regression With Prediction of Random Missing Values')

with dataset:
    test = st.file_uploader(label='Upload your dataset as pickle file', type='pkl')

# Import necessary libraries
import numpy as np
import pandas as pd
import sklearn
import importlib
import missing_values as msv
import encoding
import pickle

# Read the training data
import os
reg_dict1_file_path = os.getcwd() + '/reg_dict1.pkl'
reg_dict2_file_path = os.getcwd() + '/reg_dict2.pkl'
reg_dict3_file_path = os.getcwd() + '/reg_dict3.pkl'
reg_dict4_file_path = os.getcwd() + '/reg_dict4.pkl'
reg_dict5_file_path = os.getcwd() + '/reg_dict5.pkl'
reg_dict6_file_path = os.getcwd() + '/reg_dict6.pkl'
reg_dict7_file_path = os.getcwd() + '/reg_dict7.pkl'
reg_dict8_file_path = os.getcwd() + '/reg_dict8.pkl'
reg_dict9_file_path = os.getcwd() + '/reg_dict9.pkl'
reg_dict10_file_path = os.getcwd() + '/reg_dict10.pkl'
reg_dict11_file_path = os.getcwd() + '/reg_dict11.pkl'
reg_dict12_file_path = os.getcwd() + '/reg_dict12.pkl'

mean_mode_dict_file_path = os.getcwd() + '/train_mean_mode.pkl'
LR_file_path = os.getcwd() + '/linear_reg.pkl'
LR_simple_file_path = os.getcwd() + '/linear_reg_simple.pkl'
encoder_file_path = os.getcwd() + '/encoder.pkl'
test_file_path = os.getcwd() + '/test_file.pkl'


with open('reg_dict1.pkl', 'rb') as file:
    reg_dict1 = pickle.load(file)
with open('reg_dict2.pkl', 'rb') as file:
    reg_dict2 = pickle.load(file)
with open('reg_dict3.pkl', 'rb') as file:
    reg_dict3 = pickle.load(file)
with open('reg_dict4.pkl', 'rb') as file:
    reg_dict4 = pickle.load(file)
with open('reg_dict5.pkl', 'rb') as file:
    reg_dict5 = pickle.load(file)
with open('reg_dict6.pkl', 'rb') as file:
    reg_dict6 = pickle.load(file)
with open('reg_dict7.pkl', 'rb') as file:
    reg_dict7 = pickle.load(file)
with open('reg_dict8.pkl', 'rb') as file:
    reg_dict8 = pickle.load(file)
with open('reg_dict9.pkl', 'rb') as file:
    reg_dict9 = pickle.load(file)
with open('reg_dict10.pkl', 'rb') as file:
    reg_dict10 = pickle.load(file)
with open('reg_dict11.pkl', 'rb') as file:
    reg_dict11 = pickle.load(file)
with open('reg_dict12.pkl', 'rb') as file:
    reg_dict12 = pickle.load(file)

with open(mean_mode_dict_file_path, 'rb') as file:
    train_mean_mode = pickle.load(file)

with open(LR_file_path, 'rb') as file:
    LR = pickle.load(file)

with open(LR_simple_file_path, 'rb') as file:
    LR_simple = pickle.load(file)

with open(encoder_file_path, 'rb') as file:
    enc = pickle.load(file)

with open(test_file_path, 'rb') as file:
    X_test = pickle.load(file)



keyss = []
valuess = []
a = reg_dict1.keys()
b = reg_dict1.values()
for i in a:
    keyss.append(i)
for i in b:
    valuess.append(i)

a = reg_dict2.keys()
b = reg_dict2.values()
for i in a:
    keyss.append(i)
for i in b:
    valuess.append(i)

a = reg_dict3.keys()
b = reg_dict3.values()
for i in a:
    keyss.append(i)
for i in b:
    valuess.append(i)

a = reg_dict4.keys()
b = reg_dict4.values()
for i in a:
    keyss.append(i)
for i in b:
    valuess.append(i)

a = reg_dict5.keys()
b = reg_dict5.values()
for i in a:
    keyss.append(i)
for i in b:
    valuess.append(i)

a = reg_dict6.keys()
b = reg_dict6.values()
for i in a:
    keyss.append(i)
for i in b:
    valuess.append(i)

a = reg_dict7.keys()
b = reg_dict7.values()
for i in a:
    keyss.append(i)
for i in b:
    valuess.append(i)

a = reg_dict8.keys()
b = reg_dict8.values()
for i in a:
    keyss.append(i)
for i in b:
    valuess.append(i)

a = reg_dict9.keys()
b = reg_dict9.values()
for i in a:
    keyss.append(i)
for i in b:
    valuess.append(i)

a = reg_dict10.keys()
b = reg_dict10.values()
for i in a:
    keyss.append(i)
for i in b:
    valuess.append(i)

a = reg_dict11.keys()
b = reg_dict11.values()
for i in a:
    keyss.append(i)
for i in b:
    valuess.append(i)

a = reg_dict12.keys()
b = reg_dict12.values()
for i in a:
    keyss.append(i)
for i in b:
    valuess.append(i)

regressor_dict = {}
for i in range(len(keyss)):
    regressor_dict[keyss[i]] = valuess[i]

# if test != None:
#     X_test = test
# Preprocessing
#X_test.drop('Id',axis=1,inplace=True)
test_encoded = encoding.my_encoder(X_test, enc)

# My Immputation
test_imputed = msv.my_test_imputer(test=test_encoded, regressor_dict=regressor_dict, mean_mode=train_mean_mode)

# model prediction
y_predict = LR.predict(test_imputed)

# Save results
y_predict = pd.DataFrame(y_predict)
y_predict.to_csv('predictions.csv')

with open('predictions.csv', 'rb') as file:  
    # A new file will be created
    with predicitons:
        btn = st.download_button(
            label='Download predictions',
            file_name='predictions.csv',
            data=file,
            mime='text/csv',)