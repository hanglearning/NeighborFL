import numpy as np

""" Functions to process the data for training and predicting

NOTE - set the `attr` value for the learning and predicting feature 
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler

attr = 'Speed'

def get_scaler(df_whole):
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df_whole[attr].values.reshape(-1, 1))
    return scaler

def process_data(df_data, scaler, INPUT_LENGTH, OUTPUT_LENGTH):
    
    flow_data = scaler.transform(df_data[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    data_set = []
    
    for i in range(INPUT_LENGTH, len(flow_data) - (OUTPUT_LENGTH - 1)):
        data_set.append(flow_data[i - INPUT_LENGTH: i + OUTPUT_LENGTH])
    data = np.array(data_set) 
    
    X = data[:, :-OUTPUT_LENGTH]
    y = data[:, -OUTPUT_LENGTH:]

    return X, y 

