import numpy as np

""" Functions to process the data for training and predicting
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_scaler(df_whole, attr):
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df_whole[attr].values.reshape(-1, 1))
    return scaler

def process_train_data(df_data, scaler, INPUT_LENGTH, OUTPUT_LENGTH, attr):
    
    flow_data = scaler.transform(df_data[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    data_set = []
    
    for i in range(INPUT_LENGTH, len(flow_data) - (OUTPUT_LENGTH - 1)):
        data_set.append(flow_data[i - INPUT_LENGTH: i + OUTPUT_LENGTH])
    data = np.array(data_set) 
    
    X = data[:, :-OUTPUT_LENGTH]
    y = data[:, -OUTPUT_LENGTH:]

    return X, y  

def process_test_data(df_data, scaler, INPUT_LENGTH, OUTPUT_LENGTH, comm_round, attr):
    
    flow_data = scaler.transform(df_data[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    X, y = [], []
    
    # get y
    for i in range(INPUT_LENGTH, len(flow_data) - (OUTPUT_LENGTH - 1)):
        y.append(flow_data[i: i + OUTPUT_LENGTH])
    y = np.array(y) 

    # get X
    for i in range(len(flow_data) - INPUT_LENGTH):
        X.append(flow_data[i: i + INPUT_LENGTH])
    X = np.array(X) 

    if comm_round != 1:
        # get rid of the first (O - 1) prediction instances
        X = X[OUTPUT_LENGTH - 1:]
    return X, y  