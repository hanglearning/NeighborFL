"""
Defination of NN GRU model
"""
from keras.layers import Dense, Dropout, Activation, GRU, LSTM
from keras.models import Sequential

def create_gru(units, configs):
    """GRU(Gated Recurrent Unit)
    Build GRU Model.
    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(GRU(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(GRU(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))
    
    model.compile(loss=configs[0], optimizer=configs[1], metrics=[configs[2]])

    return model

"""
Defination of NN LSTM model
"""

def create_lstm(units, configs):
    """LSTM(Long Short-Term Memory)
    Build LSTM Model.

    # Arguments
        units: List(int), number of input, output and hidden units.
    # Returns
        model: Model, nn model.
    """

    model = Sequential()
    model.add(LSTM(units[1], input_shape=(units[0], 1), return_sequences=True))
    model.add(LSTM(units[2]))
    model.add(Dropout(0.2))
    model.add(Dense(units[3], activation='sigmoid'))

    model.compile(loss=configs[0], optimizer=configs[1], metrics=[configs[2]])
    
    return model