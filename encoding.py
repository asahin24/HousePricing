import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import OrdinalEncoder

def my_encoder(data, encoder = None):
        data_copy = copy.deepcopy(data)
        grades = ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
        literal = ['Ex', 'Gd', 'TA', 'Fa', 'Po', np.nan]
        num = [9, 7, 5, 3, 2, 1]
        G = dict(zip(literal, num))
        data_copy[grades] = data_copy[grades].replace(G)

        object_cols = data_copy.select_dtypes(include=['object']).columns
        data_encoded = copy.deepcopy(data_copy)
        if encoder == None:
                enc = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value=np.nan)
                data_encoded[object_cols] =  enc.fit_transform(data_copy[object_cols])
                return data_encoded, enc
        else:
               data_encoded[object_cols] =  encoder.transform(data_copy[object_cols]) 
        return data_encoded