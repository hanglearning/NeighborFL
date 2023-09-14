# code for individual pre-training

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

from copy import deepcopy

import random
import tensorflow as tf

# remove some warnings
import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# arguments for system vars
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="KFRT pretrain")
parser.add_argument('-dp', '--dataset_path', type=str, default=None, help='path to traffic datasets')
parser.add_argument('-sd', '--seed', type=int, default=40, help='random seed value specified for reproducibility')
parser.add_argument('-m', '--model', type=str, default='lstm', help='choose LSTM or GRU model for Central/Federated simulations. add new model structures in create_model.py')
parser.add_argument('-I', '--input_length', type=int, default=12, help='input length for the LSTM/GRU network')
parser.add_argument('-O', '--output_length', type=int, default=1, help='output length for the LSTM/GRU network')
parser.add_argument('-hn', '--hidden_neurons', type=int, default=128, help='number of neurons in one of each 2 layers')
parser.add_argument('-lo', '--loss', type=str, default="mse", help='loss function used for training')
parser.add_argument('-op', '--optimizer', type=str, default="rmsprop", help='optimizer for training')
parser.add_argument('-me', '--metric', type=str, default="mse", help='evaluation metric by the model')
parser.add_argument('-b', '--batch', type=int, default=1, help='batch number for training')
parser.add_argument('-e', '--epochs', type=int, default=5, help='pretrain epoch number')
parser.add_argument('-pp', '--pretrain_percent', type=float, default=1.0, help='percentage of the data for pretraining')
parser.add_argument('-si', '--pretrain_start_index', type=int, default=0, help='the starting row for the pretrained models')
parser.add_argument('-ei', '--pretrain_end_index', type=int, default=None, help='till which row in df we do pretrain. if this is provide, overwrite -pp')
parser.add_argument('-f', '--feature', type=str, default='Speed', help='feature used for training and prediction, depending on your traffic dataset, usually speed, volume, occupancy')
parser.add_argument('-sp', '--model_save_path', type=str, default=None, help='the path to save the pretrained models')

args = parser.parse_args()
args = args.__dict__

def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(args["seed"])
   tf.random.set_seed(args["seed"])
   np.random.seed(args["seed"])
   random.seed(args["seed"])

reset_random_seeds()

# init default dataset folder
if not args["dataset_path"]:
    args["dataset_path"] = f"{os.getcwd()}/data/"

if not args["model_save_path"]:
    args["model_save_path"] = f"{os.getcwd()}/pretrained_models/"
os.makedirs(args["model_save_path"], exist_ok=True)

# read in device file paths
all_device_files = [f for f in listdir(args["dataset_path"]) if isfile(join(args["dataset_path"], f)) and '.csv' in f and not 'location' in f]
print(f'We have {len(all_device_files)} devices available.')

# init first model

create_model = create_lstm if args["model"] == 'lstm' else create_gru
model_units = [args['input_length'], args['hidden_neurons'], args['hidden_neurons'], args['output_length']]
model_configs = [args['loss'], args['optimizer'], args['metric']]
global_model_0 = create_model(model_units, model_configs)

''' save used arguments as a text file for easy review '''
with open(f'{args["model_save_path"]}/args_used.txt', 'w') as f:
    f.write("Command line arguments used -\n")
    f.write(' '.join(sys.argv[1:]))
    f.write("\n\nAll arguments used -\n")
    for arg_name, arg in args.items():
        f.write(f'\n--{arg_name} {arg}')
    f.write("\nThis file is mainly used to recall the generated model structure.")


''' load data for each device '''
min_read_line_num = float('inf')
for device_file_iter in range(len(all_device_files)):
    device_file_name = all_device_files[device_file_iter]
    file_path = os.path.join(args['dataset_path'], device_file_name)
    
    # count lines to later determine max comm rounds
    whole_data = pd.read_csv(file_path, encoding='utf-8').fillna(0)
    read_to_line = int(whole_data.shape[0] * args["pretrain_percent"])
    if args["pretrain_end_index"]:
        read_to_line = args["pretrain_end_index"]
    min_read_line_num = min(min_read_line_num, read_to_line)

whole_data_record = {} # to calculate scaler
for device_file_iter in range(len(all_device_files)):
    device_file_name = all_device_files[device_file_iter]
    sensor_id = device_file_name.split('-')[-1].strip().split(".")[0]
    # data file path
    file_path = os.path.join(args['dataset_path'], device_file_name)
    
    # count lines to later determine max comm rounds
    whole_data = pd.read_csv(file_path, encoding='utf-8').fillna(0)
    whole_data = whole_data[:min_read_line_num]
    print(f'Loaded {min_read_line_num} lines of data from {device_file_name}. Last training timestamp {whole_data.iloc[min_read_line_num - 1]["Timestamp"]}. ({device_file_iter+1}/{len(all_device_files)})')
    whole_data_record[sensor_id] = whole_data

    ''' get scaler '''
    scaler = get_scaler(pd.concat(list(whole_data_record.values())), args['feature'])

sample_sensor_id = list(whole_data_record.keys())[0]
start_timestamp = whole_data_record[sample_sensor_id].iloc[args["pretrain_start_index"]]['Timestamp']
end_timestamp = whole_data_record[sample_sensor_id].iloc[args["pretrain_end_index"] - 1]['Timestamp']
print(f"Pretraining starts with {start_timestamp} and end with {end_timestamp} (inclusive).")

for sensor_id, data in whole_data_record.items():
    ''' Process traning data '''
    # process training data
    X_train, y_train = process_train_data(data, scaler, args['input_length'], args['output_length'], args['feature'])

    print(f"{sensor_id} pretraining")
    init_model = deepcopy(global_model_0)
    new_model = train_model(init_model, X_train, y_train, args['batch'], args['epochs'])
    new_model.save(f"{args['model_save_path']}}/{sensor_id}.h5")

print(f"Pretrain done till line {min_read_line_num} for each device.")