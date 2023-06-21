# code for individual pre-training
import sys, os
sys.path.append(os.getcwd())

from Detector import Detector

from os import listdir
from os.path import isfile, join
import csv
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import argparse

from create_model import create_lstm
from create_model import create_gru
from model_training import train_model

from process_data import get_scaler
from process_data import process_train_data
from process_data import process_test_one_step
from process_data import process_test_multi_and_get_y_true

from copy import deepcopy

import random
import tensorflow as tf

# remove some warnings
import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# arguments for system vars
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="offline FLTP training to get a global model")
parser.add_argument('-dp', '--dataset_path', type=str, default='/content/drive/MyDrive/Hang_PeMS/PeMS-Bay 061223/PeMS-Bay Selected/csv', help='dataset path')
parser.add_argument('-sd', '--seed', type=int, default=40, help='random seed for reproducibility')
parser.add_argument('-m', '--model', type=str, default='lstm', help='Model to choose - lstm or gru')
parser.add_argument('-il', '--input_length', type=int, default=12, help='input length for the LSTM/GRU network')
parser.add_argument('-hn', '--hidden_neurons', type=int, default=128, help='number of neurons in one of each 2 layers')
parser.add_argument('-lo', '--loss', type=str, default="mse", help='loss evaluation while training')
parser.add_argument('-op', '--optimizer', type=str, default="rmsprop", help='optimizer for training')
parser.add_argument('-me', '--metrics', type=str, default="mse", help='evaluation metrics to view for training')
parser.add_argument('-b', '--batch', type=int, default=1, help='batch number for training')
parser.add_argument('-e', '--epochs', type=int, default=5, help='pretrain epoch number')
parser.add_argument('-si', '--train_start_index', type=int, default=0, help='the starting row for the offline FLTP global model')
parser.add_argument('-ei', '--train_end_index', type=int, default=0, help='end index of offline training')
parser.add_argument('-sp', '--model_save_path', type=str, default="/content/drive/MyDrive/Hang_PeMS/PeMS-Bay 061223/PeMS-Bay Selected/offline_FLTP_model", help='the path to save the offline FLTP global model')
parser.add_argument('-cr', '--comm_rounds', type=int, default=5, help='number of communication rounds')

args = parser.parse_args()
args = args.__dict__

# read in detector file paths
all_detector_files = [f for f in listdir(args["dataset_path"]) if isfile(join(args["dataset_path"], f)) and '.csv' in f and not 'location' in f]
print(f'We have {len(all_detector_files)} detectors available.')

# init first model

create_model = create_lstm if args["model"] == 'lstm' else create_gru
model_units = [args['input_length'], args['hidden_neurons'], args['hidden_neurons'], 1]
model_configs = [args['loss'], args['optimizer'], args['metrics']]
global_model_0 = create_model(model_units, model_configs)

''' load data for each detector '''
min_read_line_num = float('inf')
for detector_file_iter in range(len(all_detector_files)):
    detector_file_name = all_detector_files[detector_file_iter]
    file_path = os.path.join(args['dataset_path'], detector_file_name)
    
    # count lines to later determine max comm rounds
    whole_data = pd.read_csv(file_path, encoding='utf-8').fillna(0)
    read_to_line = args["train_end_index"]
    min_read_line_num = min(min_read_line_num, read_to_line)

whole_data_record = {} # to calculate scaler
for detector_file_iter in range(len(all_detector_files)):
    detector_file_name = all_detector_files[detector_file_iter]
    sensor_id = detector_file_name.split('-')[-1].strip().split(".")[0]
    # data file path
    file_path = os.path.join(args['dataset_path'], detector_file_name)
    
    # count lines to later determine max comm rounds
    whole_data = pd.read_csv(file_path, encoding='utf-8').fillna(0)
    whole_data = whole_data[:min_read_line_num]
    print(f'Loaded {min_read_line_num} lines of data from {detector_file_name}. Last training timestamp {whole_data.iloc[min_read_line_num - 1]["Timestamp"]}. ({detector_file_iter+1}/{len(all_detector_files)})')
    whole_data_record[sensor_id] = whole_data

    ''' get scaler '''
    scaler = get_scaler(pd.concat(list(whole_data_record.values())))

sample_sensor_id = list(whole_data_record.keys())[0]
start_timestamp = whole_data_record[sample_sensor_id].iloc[args["train_start_index"]]['Timestamp']
end_timestamp = whole_data_record[sample_sensor_id].iloc[args["train_end_index"] - 1]['Timestamp']
print(f"Offline FLTP training starts with {start_timestamp} and end with {end_timestamp} (inclusive).")

# init models
sensor_id_to_model = {}
for sensor_id in whole_data_record:
    sensor_id_to_model[sensor_id] = deepcopy(global_model_0)

# process training data
sensor_id_to_data = {}
for sensor_id, data in whole_data_record.items():    
    X_train, y_train = process_train_data(data, scaler, args['input_length'])
    sensor_id_to_data[sensor_id] = (X_train, y_train)

# Offline FL starts
all_sensor_ids = whole_data_record.keys()
for round in range(1, args['comm_rounds'] + 1):
    print(f"Comm round {round}...")
    local_models_weights = []
    for ind, sensor_id in enumerate(list(all_sensor_ids)):
        print(f"{sensor_id} training ({ind + 1}/{len(all_sensor_ids)})")
        model = train_model(sensor_id_to_model[sensor_id], sensor_id_to_data[sensor_id][0], sensor_id_to_data[sensor_id][1], args['batch'], args['epochs'])
        sensor_id_to_model[sensor_id] = model
        local_models_weights.append(model.get_weights())
    global_model = create_model(model_units, model_configs)
    global_model.set_weights(np.mean(local_models_weights, axis=0))
    # update each sensor's model
    for ind, sensor_id in enumerate(list(all_sensor_ids)):
        sensor_id_to_model[sensor_id] = global_model

global_model.save(f"{args['model_save_path']}/global_model.h5")
print(f"Offline FLTP done for {args['comm_rounds']} comm rounds with {args['epochs']} local epochs till line {min_read_line_num} for each detector.")

