import numpy as np
import pandas as pd
import sklearn
import copy
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model

def get_relevant_feats(data, relation_threshold = 0.4):
#Using Pearson Correlations
    cor = data.corr()
    relevant_feats_dict = {}
    for col in data.columns:
        #Correlation with output variable
        cor_target = abs(cor[col])
        #Selecting highly correlated features
        relevant_feats = cor_target[cor_target > relation_threshold]
        relevant_feats.drop(col,inplace=True)
        relevant_feats_dict[col] = relevant_feats
    return relevant_feats_dict

def get_RF_dict(data, relevant_feats = None):

    RF_dict = {}
    if relevant_feats == None:
        columns = data.columns
    else:
        columns = relevant_feats
    for col in columns:
        target = data[col]
        data_train = data.drop(col, axis=1)
        RF = RandomForestRegressor()
        RF.fit(data_train, target)
        RF_dict[col] = RF
    return RF_dict

def get_LR_dict(data, relevant_feats = None):

    LR_dict = {}
    if relevant_feats == None:
        columns = data.columns
    else:
        columns = relevant_feats
    for col in columns:
        target = data[col]
        data_train = data.drop(col, axis=1)
        LR = linear_model.Lasso(alpha=0.2)
        LR.fit(data_train, target)
        LR_dict[col] = LR
    return LR_dict
    
def get_mean_mode(data):
    mean_mode = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            mean_mode[col] = data[col].mode()[0]
        else:
            mean_mode[col] = data[col].mean()
    return mean_mode

def simple_imputer(data, mean_mode = None):
    data_copy = copy.deepcopy(data)
    if mean_mode == None:
        mean_mode = get_mean_mode(data)
        for col,_ in data_copy.items():
            if data_copy[col].dtype == 'object':
                data_copy[col].fillna(data_copy[col].mode()[0], inplace = True)
            else:
                data_copy[col].fillna(data_copy[col].mean(), inplace = True)
        return data_copy, mean_mode
    else:
        for col in data_copy.columns:
            data_copy[col].fillna(mean_mode[col], inplace = True)
        return data_copy

def my_train_imputer(data, relation_threshold = None, regressor = 'RF'):
    data_copy = copy.deepcopy(data)
    columns_with_null = data_copy.columns[data_copy.isna().any()].tolist()

    data_copy = copy.deepcopy(data)
    data_imputed, mean_mode = simple_imputer(data_copy)
    relevant_feats = get_relevant_feats(data_imputed, relation_threshold)
    if regressor == 'RF':
        regressor_dict = get_RF_dict(data_imputed, relevant_feats)
    else:
        regressor_dict = get_LR_dict(data_imputed, relevant_feats)

    for i in data_imputed.index:
        row = copy.deepcopy(data.loc[i,:])
        for col in data_copy.columns:
            if math.isnan(data_copy.loc[i,col]):
                data_copy.loc[i,col] = regressor_dict[col].predict(pd.Series.to_frame(data_imputed.drop(col, axis=1).loc[i,:]).transpose())

    return data_copy, columns_with_null, regressor_dict, mean_mode

def my_test_imputer(test, regressor_dict, mean_mode):
    test_copy = copy.deepcopy(test)
    test_imputed = simple_imputer(data=test_copy, mean_mode=mean_mode)
    for i in test_imputed.index:
        row = copy.deepcopy(test_copy.loc[i,:])
        for col in test_copy.columns:
            if math.isnan(test_copy.loc[i,col]):
                test_copy.loc[i,col] = regressor_dict[col].predict(pd.Series.to_frame(test_imputed.drop(col, axis=1).loc[i,:]).transpose())
    return test_copy