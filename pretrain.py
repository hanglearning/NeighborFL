# code for individual pre-training
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
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="KFRT pretrain")
parser.add_argument('-dp', '--dataset_path', type=str, default='/content/drive/MyDrive/KFRT_data', help='dataset path')
parser.add_argument('-sd', '--seed', type=int, default=40, help='random seed for reproducibility')
parser.add_argument('-m', '--model', type=str, default='lstm', help='Model to choose - lstm or gru')
parser.add_argument('-il', '--input_length', type=int, default=12, help='input length for the LSTM/GRU network')
parser.add_argument('-hn', '--hidden_neurons', type=int, default=128, help='number of neurons in one of each 2 layers')
parser.add_argument('-lo', '--loss', type=str, default="mse", help='loss evaluation while training')
parser.add_argument('-op', '--optimizer', type=str, default="rmsprop", help='optimizer for training')
parser.add_argument('-me', '--metrics', type=str, default="mape", help='evaluation metrics for training')
parser.add_argument('-b', '--batch', type=int, default=1, help='batch number for training')
parser.add_argument('-e', '--epochs', type=int, default=5, help='epoch number per comm comm_round for FL')
parser.add_argument('-pp', '--pretrain_percent', type=float, default=0.5, help='percentage of the data for pretraining')

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
    read_to_line = int(whole_data.shape[0] * args["pretrain_percent"])
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
    print(f'Loaded {min_read_line_num} lines of data from {detector_file_name} (percentage: {args["pretrain_percent"]}). ({detector_file_iter+1}/{len(all_detector_files)})')
    whole_data_record[sensor_id] = whole_data

    ''' get scaler '''
    scaler = get_scaler(pd.concat(list(whole_data_record.values())))

for sensor_id, data in whole_data_record.items():
    ''' Process traning data '''
    # process training data
    X_train, y_train = process_train_data(data, scaler, args['input_length'])

    print(f"{sensor_id} pretraining")
    init_model = deepcopy(global_model_0)
    new_model = train_model(init_model, X_train, y_train, args['batch'], args['epochs'])
    new_model.save(f"/content/drive/MyDrive/Hang_PeMS/pretrained_models/{sensor_id}.h5")

print(f"Pretrain till line {min_read_line_num}")