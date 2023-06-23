from Detector import Detector

import os
from os import listdir
from os.path import isfile, join
import sys
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import argparse

from process_data import get_scaler
from process_data import process_train_data
from process_data import process_test_one_step
from process_data import process_test_multi_and_get_y_true

import random
import tensorflow as tf

import shutil

from tensorflow.keras.models import load_model


# remove some warnings
import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

''' Parse command line arguments '''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Offline FLTP Prediction")

# arguments for system vars
parser.add_argument('-mp', '--model_path', type=str, default='/content/drive/MyDrive/Hang_NeighborFL/PeMS-Bay 061223/PeMS-Bay Selected/offline_FLTP_model/global_model.h5', help='The path of global model for every detector')
parser.add_argument('-dp', '--dataset_path', type=str, default='/content/drive/MyDrive/Hang_NeighborFL/PeMS-Bay 061223/PeMS-Bay Selected/csv', help='dataset path')
parser.add_argument('-lb', '--logs_base_folder', type=str, default="/content/drive/MyDrive/Hang_NeighborFL/NeighborFL_logs", help='base folder path to store running logs and h5 files')
parser.add_argument('-sd', '--seed', type=int, default=40, help='random seed for reproducibility')

# arguments for prediction
parser.add_argument('-il', '--input_length', type=int, default=12, help='input length for the LSTM/GRU network')
parser.add_argument('-ff', '--num_feedforward', type=int, default=12, help='number of feedforward predictions, used to set up the number of the last layer of the model (usually it has to be equal to -il)')
parser.add_argument('-si', '--start_index', type=int, default=0, help='the index to start taking data for prediction')
parser.add_argument('-ms', '--max_data_size', type=int, default=72, help='maximum data length for training in each communication comm_round, simulating the memory space a detector has') # only used on Comm round 1

# arguments for federated learning
parser.add_argument('-c', '--comm_rounds', type=int, default=335, help='number of comm rounds, default aims to run until data is exhausted')

args = parser.parse_args()
args = args.__dict__

# https://stackoverflow.com/questions/60058588/tesnorflow-2-0-tf-random-set-seed-not-working-since-i-am-getting-different-resul
def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(args["seed"])
   tf.random.set_seed(args["seed"])
   np.random.seed(args["seed"])
   random.seed(args["seed"])

reset_random_seeds()

INPUT_LENGTH = args['input_length']
new_sample_size_per_comm_round = INPUT_LENGTH

# create log folder indicating by current running date and time
date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
logs_dirpath = f"{args['logs_base_folder']}/{date_time}_Offline_FLTP_Prediction"
os.makedirs(f"{logs_dirpath}", exist_ok=True)
print(f"Predictions stored in {logs_dirpath}")

all_detector_files = [f for f in listdir(args["dataset_path"]) if isfile(join(args["dataset_path"], f)) and '.csv' in f and not 'location' in f]
print(f'We have {len(all_detector_files)} detectors available.')

detector_predicts = {}
for detector_file in all_detector_files:
    sensor_id = detector_file.split('-')[-1].strip().split(".")[0]
    detector_predicts[sensor_id] = {}
    detector_predicts[sensor_id]['offline_fl'] = []
    # true
    detector_predicts[sensor_id]['true'] = []
    
''' load data for each detector '''
whole_data_record = {} # to calculate scaler
for detector_file_iter in range(len(all_detector_files)):
    detector_file_name = all_detector_files[detector_file_iter]
    sensor_id = detector_file_name.split('-')[-1].strip().split(".")[0]
    # data file path
    file_path = os.path.join(args['dataset_path'], detector_file_name)
    
    whole_data = pd.read_csv(file_path, encoding='utf-8').fillna(0)
    
    print(f'Loaded {whole_data.shape[0]} lines of data from {detector_file_name}. ({detector_file_iter+1}/{len(all_detector_files)})')
    whole_data_record[sensor_id] = whole_data

scaler = get_scaler(pd.concat(list(whole_data_record.values())))
global_model = load_model(args["model_path"], compile = False)

run_comm_rounds = args["comm_rounds"]

for comm_round in range(1, run_comm_rounds):
    print(f"Simulating comm comm_round {comm_round}/{run_comm_rounds} ({comm_round/run_comm_rounds:.0%})...")
    ''' calculate simulation data range '''
    # train data
    if comm_round == 1:
        training_data_starting_index = args["start_index"]
        training_data_ending_index = training_data_starting_index + new_sample_size_per_comm_round * 2 - 1
        # if it's comm_round 1 and input_shape 12, need at least 24 training data points because the model at least needs 13 points to train.
        # Therefore,
        # comm_round 1 -> 1~24 training points, predict with test 13~35 test points, 
        # 1- 24， 2 - 36， 3 - 48， 4 - 60
    else:
        training_data_ending_index = args["start_index"] + (comm_round + 1) * new_sample_size_per_comm_round - 1
        training_data_starting_index = max(args["start_index"], training_data_ending_index - args['max_data_size'] + 1)
    # test data
    test_data_starting_index = training_data_ending_index - new_sample_size_per_comm_round + 1
    test_data_ending_index_one_step = test_data_starting_index + new_sample_size_per_comm_round * 2 - 1
    test_data_ending_index_chained_multi = test_data_starting_index + new_sample_size_per_comm_round - 1
    
    detecotr_iter = 1
    for sensor_id in whole_data_record:
        ''' Process traning data '''
        if comm_round == 1:
            # special dealing
            # slice training data
            train_data = whole_data_record[sensor_id][training_data_starting_index: training_data_ending_index + 1]
            # process training data
            X_train, y_train = process_train_data(train_data, scaler, INPUT_LENGTH)
        ''' Process test data '''
        # slice test data
        test_data = whole_data_record[sensor_id][test_data_starting_index: test_data_ending_index_one_step + 1]
        # process test data
        X_test, _ = process_test_one_step(test_data, scaler, INPUT_LENGTH)
        _, y_true = process_test_multi_and_get_y_true(test_data, scaler, INPUT_LENGTH, args['num_feedforward'])
        # reshape data
        for data_set in ['X_train', 'X_test']:
            vars()[data_set] = np.reshape(vars()[data_set], (vars()[data_set].shape[0], vars()[data_set].shape[1], 1))
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(1, -1)[0]
        if comm_round == 1:
            # special dealing
            y_train = scaler.inverse_transform(y_train.reshape(-1, 1)).reshape(1, -1)[0]
            detector_predicts[sensor_id]['true'].append((1,y_train))
            offline_predictions = global_model.predict(X_train)
            offline_predictions = scaler.inverse_transform(offline_predictions.reshape(-1, 1)).reshape(1, -1)[0]
            detector_predicts[sensor_id]['offline_fl'].append((1,offline_predictions))
        
        ''' Predict '''
        detector_predicts[sensor_id]['true'].append((comm_round + 1,y_true))
        print(f"{sensor_id} is now predicting by offline FLTP global model...")
        offline_predictions = global_model.predict(X_test)
        offline_predictions = scaler.inverse_transform(offline_predictions.reshape(-1, 1)).reshape(1, -1)[0]
        detector_predicts[sensor_id]['offline_fl'].append((comm_round + 1,offline_predictions))

        detecotr_iter += 1
    
    print(f"Saving progress for comm_round {comm_round} and predictions in comm_round {comm_round + 1}...")
    
    print("Saving Predictions...")                          
    predictions_record_saved_path = f'{logs_dirpath}/all_detector_predicts.pkl'
    with open(predictions_record_saved_path, 'wb') as f:
        pickle.dump(detector_predicts, f)

print(f"Simulation done for specified comm rounds. \nIn this simulation, the last run comm rounds should be {comm_round - 1}. This is not a surprise due to the code structure that makes predictions for round n + 1 in round n.")



    