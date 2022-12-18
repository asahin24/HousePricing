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
import missing_values as msv
import encoding
import pickle

# Read the training data
import os
regressor_dict_file_path = os.getcwd() + '/regressor_dict.pkl'
mean_mode_dict_file_path = os.getcwd() + '/train_mean_mode.pkl'
LR_file_path = os.getcwd() + '/linear_reg.pkl'
LR_simple_file_path = os.getcwd() + '/linear_reg_simple.pkl'
encoder_file_path = os.getcwd() + '/encoder.pkl'
test_file_path = os.getcwd() + '/test_file.pkl'


with open(regressor_dict_file_path, 'rb') as file:
    regressor_dict = pickle.load(file)

with open(mean_mode_dict_file_path, 'rb') as file:
    train_mean_mode = pickle.load(file)

with open(LR_file_path, 'rb') as file:
    LR = pickle.load(file)

with open(LR_simple_file_path, 'rb') as file:
    LR_simple = pickle.load(file)

with open(encoder_file_path, 'rb') as file:
    enc = pickle.load(file)

with open(test_file_path, 'rb') as file:
    # A new file will be created
    X_test = pickle.load(file)

if test != None:
    X_test = test
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