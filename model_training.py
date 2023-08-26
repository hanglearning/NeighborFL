from keras.models import Model
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model
import os

def train_model(model, X_train, y_train, batch, epochs):
    """Model training

    # Arguments
        model: model object
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), label data for train.
        batch, epochs: configurable parameter for train.
    """
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    hist = model.fit(
        X_train, y_train,
        batch_size=batch,
        epochs=epochs,
        validation_split=0.00)
    return model
