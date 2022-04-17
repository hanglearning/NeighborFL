""" Model training functions for

(1) baseline model - a model that is trained solely by the device's own data without federation, simulating a centralized learning. Used to compare with federated models.

(2) local model - the local model defined in FL.
"""

# Prepare for TF training function
"""
Train the NN model.
"""

from keras.models import Model
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model
from build_lstm import build_lstm
from build_gru import build_gru
import os

def train_model(model, X_train, y_train, batch, epochs):
    """train the baseline model 

    # Arguments
        comm_comm_round: FL communication comm_round number
        model_path: model weights to load to continue training
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), label data for train.
        sensor_id: id of the sensor, e.g., 19985_NB
        this_sensor_dirpath: specify directory to store related records for this sensor
        config: Dict, parameter for train.
    """
    #model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    hist = model.fit(
        X_train, y_train,
        batch_size=batch,
        epochs=epochs,
        validation_split=0.00)
    return model
