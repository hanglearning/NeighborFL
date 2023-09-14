
from Device import Device

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
from process_data import process_test_data

from error_calc import get_MAE
from error_calc import get_MSE
from error_calc import get_RMSE
from error_calc import get_MAPE

import random
import tensorflow as tf

import shutil

# remove TF warnings
import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

''' Parse command line arguments '''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="NeighborFL Simulation")
# arguments for system vars
parser.add_argument('-dp', '--dataset_path', type=str, default=None, help='path to traffic datasets')
parser.add_argument('-lb', '--logs_base_folder', type=str, default=None, help='base folder path to store running logs and h5 files')
parser.add_argument('-pm', '--preserve_historical_models', type=int, default=1, help='whether to preserve models from old communication rounds. Consume storage. Input 1 to preserve')
parser.add_argument('-sd', '--seed', type=int, default=40, help='random seed value specified for reproducibility')

# arguments for resume training
parser.add_argument('-rp', '--resume_path', type=str, default=None, help='provide the interrupted log folder path to continue the simulation. note that the program will print out this folder path upon executing')

# arguments for Central/Federated Real-time Learning
parser.add_argument('-c', '--comm_rounds', type=int, default=None, help='number of comm rounds, default aims to run until data is exhausted')
parser.add_argument('-ms', '--max_data_size', type=int, default=72, help='maximum data length for training in each communication round, simulating the memory size the devices have')
parser.add_argument('-f', '--feature', type=str, default='Speed', help='feature used for training and prediction, depending on your traffic dataset, usually speed, volume, occupancy')
parser.add_argument('-tau1', '--tau1', type=int, default=24, help='number of data point collections and prediction amount in round 1, rule of thumb: 2 * tau2p')
parser.add_argument('-tau2p', '--tau2p', type=int, default=12, help='Number of data points collected, and also the number of predictions made in round 2 and beyond, rule of thumb - same as -I') # a better choice maybe t1 = t2 = 2 * -I
parser.add_argument('-lm', '--learning_methods', type=str, default=1111, help='Enable learning methods for simulation. Use `1` to enable and `0` to disable. The indices correspond to: 1. Baseline centralized, 2. Pure FedAvg NaiveFL, 3. RadiusFL, 4. NeighborFL')

# arguments for learning
parser.add_argument('-m', '--model', type=str, default='lstm', help='choose LSTM or GRU model for Central/Federated simulations. add new model structures in create_model.py')
parser.add_argument('-I', '--input_length', type=int, default=12, help='input length for the LSTM/GRU network')
parser.add_argument('-O', '--output_length', type=int, default=1, help='number of the prediction horizon, used to set up the number of the last layer of the model')
parser.add_argument('-hn', '--hidden_neurons', type=int, default=128, help='number of neurons in one of each 2 layers')
parser.add_argument('-lo', '--loss', type=str, default="mse", help='loss function used for training')
parser.add_argument('-op', '--optimizer', type=str, default="rmsprop", help='optimizer for training')
parser.add_argument('-me', '--metric', type=str, default="mse", help='evaluation metric by the local model during training and testing')
parser.add_argument('-b', '--batch', type=int, default=1, help='batch number for training')
parser.add_argument('-e', '--epochs', type=int, default=5, help='epoch number per comm_round for Central/FL')
parser.add_argument('-dlp', '--data_load_percent', type=float, default=1.0, help='percentage of the data to load. may overwrite -c.')
parser.add_argument('-si', '--start_train_index', type=int, default=0, help='start the simulation at this data index, usually to accommodate pretrained models')
parser.add_argument('-pp', '--pretrained_models_path', type=str, default=None, help='pretrained models path. Pretrained models path. If not provided, the program will create an initial model for all devices and train from scratch')

# arguments for NeighborFL
parser.add_argument('-r', '--radius', type=float, default=1, help='Specifies the radius within which participants are treated as candidate favorite neighbors. Unit - mile.')
parser.add_argument('-et', '--error_type', type=str, default="MSE", help='the error type to evaluate potential neighbors')
parser.add_argument('-nt', '--num_neighbors_try', type=int, default=1, help='how many new neighbors to evaluate in each comm_round')
parser.add_argument('-ah', '--add_heuristic', type=int, default=1, help='heuristic to add fav neighbors: 1 - add by distance from close to far, 2 - add randomly')
# arguments for NeighborFL removing
parser.add_argument('-rt', '--remove_trigger', type=int, default=2, help='to trigger a remove: 0 - never remove; 1 - trigger by probability, set by -ep; 2 - trigger by a consecutive rounds of error increase, set by -nu')
parser.add_argument('-ep', '--epsilon', type=float, default=0.2, help='if -rt 1, device has a probability to remove out worst neighbors to explore new neighbors')
parser.add_argument('-nu', '--remove_rounds', type=int, default=3, help="a remove will be triggered if the error of the device's NeighborFL agg model has been increasing for -nu number of rounds. if -nu 1, a removal is triggered as soon as the error increases")
parser.add_argument('-rn', '--remove_num', type=int, default=1, help='this number defines how many favorite neighbors to remove having the worst reputation.')
parser.add_argument('-rs', '--remove_strategy', type=int, default=1, help='1 - remove by worst reputation; 2 - remove randomly; 3 - always remove the last added one')

args = parser.parse_args()
args = args.__dict__

# https://stackoverflow.com/questions/60058588/tesnorflow-2-0-tf-random-set-seed-not-working-since-i-am-getting-different-resul
def reset_random_seeds():
   os.environ['PYTHONHASHSEED']=str(args["seed"])
   tf.random.set_seed(args["seed"])
   np.random.seed(args["seed"])
   random.seed(args["seed"])

reset_random_seeds()

''' Parse command line arguments '''

# centralized baseline
central_model_path = 'central'

# pure FedAvg
naive_fl_local_model_path = 'naive_fl_local'
naive_fl_global_model_path = 'naive_fl_global'

# radius FedAvg
radius_naive_fl_local_model_path = 'radius_naive_fl_local'
radius_naive_fl_agg_model_path = 'radius_naive_fl_agg'

# NeighborFL
neighbor_fl_local_model_path = 'neighbor_fl_local'
neighbor_fl_agg_model_path = 'neighbor_fl_agg'
eval_neighbor_fl_agg_model_path = 'eval_neighbor_fl_agg'

# init default dataset folder if not specified
if not args["dataset_path"]:
    args["dataset_path"] = f"{os.getcwd()}/NeighborFL/data/"

# init default log root folder if not specified
if not args["logs_base_folder"]:
    args["logs_base_folder"] = f"{os.getcwd()}/NeighborFL/logs/"

print("Preparing - i.e., create device objects, init models, load data, etc,.\nThis may take a few minutes...")
# determine if resume training
if args['resume_path']:
    logs_dirpath = args['resume_path']
    Device.logs_dirpath = logs_dirpath
    # load saved variables and see if there are arguments to overwrite
    with open(f"{logs_dirpath}/check_point/config_vars.pkl", 'rb') as f:
        config_vars = pickle.load(f)
        config_vars['resume_path'] = logs_dirpath
        # confirm overwritten args
        diff_args = {}
        # get only the command line args - https://stackoverflow.com/questions/67955098/python-argument-parser-parse-only-from-console
        NOTHING = object()
        mask = argparse.Namespace(**{arg: NOTHING for arg in args})
        masked_namespace = parser.parse_args(namespace=mask)
        command_line_args = {
            arg: value
            for arg, value in vars(masked_namespace).items()
            if value is not NOTHING
        }
        for arg in command_line_args:
            if command_line_args[arg] != config_vars[arg]:
                diff_args[arg] = command_line_args[arg]
        if diff_args:
            if 'dataset_path' in diff_args:
                print("Note, dataset is configured during the initial run and won't be overwritten, even if you specify a new/different dataset_path.")
            print("Please confirm the following args to overwrite:")
            for arg in diff_args:
                print(f"{arg}: {config_vars[arg]} -> {command_line_args[arg]}")
            if_overwrite = input("Overwrite all of them? (*/N - enter 'N' to NOT overwrite, or any other key to confirm overwrite.)")
            if if_overwrite != 'N':
                for arg in diff_args:
                    config_vars[arg] = diff_args[arg]
    with open(f'{logs_dirpath}/check_point/all_device_predicts.pkl', 'rb') as f:
        device_predicts = pickle.load(f)
    with open(f'{logs_dirpath}/check_point/fav_neighbors_by_round.pkl', 'rb') as f:
        device_TO_fav_neighbors_BY_round = pickle.load(f)
    with open(f"{logs_dirpath}/check_point/list_of_devices.pkl", 'rb') as f:
        list_of_devices = pickle.load(f)
    with open(f"{logs_dirpath}/check_point/whole_data_record.pkl", 'rb') as f:
        whole_data_record = pickle.load(f)
    STARTING_COMM_ROUND = config_vars["resume_comm_round"]
    scaler = config_vars["scaler"]
    individual_min_data_sample = config_vars["individual_min_data_sample"]
    device_locations = config_vars["device_locations"]
    
    create_model = create_lstm if config_vars["model"] == 'lstm' else create_gru
    
    model_units = [config_vars['input_length'], config_vars['hidden_neurons'], config_vars['hidden_neurons'], config_vars['output_length']]
    model_configs = [config_vars['loss'], config_vars['optimizer'], config_vars['metric']]
else:
    ''' logistics '''
    config_vars = args
    STARTING_COMM_ROUND = 1
    
    # read in device file paths
    all_device_files = [f for f in listdir(config_vars["dataset_path"]) if isfile(join(config_vars["dataset_path"], f)) and '.csv' in f and not 'location' in f]
    print(f'We have {len(all_device_files)} devices available.')
    # read in device location file
    device_location_file = [f for f in listdir(config_vars["dataset_path"]) if isfile(join(config_vars["dataset_path"], f)) and 'location' in f][0]
    device_locations = pd.read_csv(os.path.join(config_vars['dataset_path'], device_location_file))
    
    config_vars["device_locations"] = device_locations

    # create log folder indicating by current running date and time
    date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    logs_dirpath = f"{config_vars['logs_base_folder']}/{date_time}_{config_vars['model']}_mds_{config_vars['max_data_size']}_epoch_{config_vars['epochs']}_ks_{config_vars['remove_strategy']}_I_{config_vars['input_length']}_O_{config_vars['output_length']}_lm_{config_vars['learning_methods']}"
    os.makedirs(f"{logs_dirpath}/check_point", exist_ok=True)
    Device.logs_dirpath = logs_dirpath

    ''' create device object and load data for each device '''
    whole_data_record = {} # to calculate scaler
    individual_min_data_sample = float('inf') # to determine max comm rounds
    list_of_devices = {}
    for device_file_iter in range(len(all_device_files)):
        device_file_name = all_device_files[device_file_iter]
        sensor_id = device_file_name.split('-')[-1].strip().split(".")[0]
        # data file path
        file_path = os.path.join(config_vars['dataset_path'], device_file_name)
        
        # count lines to later determine max comm rounds
        whole_data = pd.read_csv(file_path, encoding='utf-8').fillna(0)
        read_to_line = int(whole_data.shape[0] * config_vars["data_load_percent"])
        individual_min_data_sample = read_to_line if read_to_line < individual_min_data_sample else individual_min_data_sample
        
        whole_data = whole_data[:read_to_line]
        print(f'Loaded {read_to_line} lines of data from {device_file_name} (percentage: {config_vars["data_load_percent"]}). ({device_file_iter+1}/{len(all_device_files)})')
        whole_data_record[sensor_id] = whole_data
        # create a device object
        device = Device(sensor_id, radius=config_vars['radius'],latitude=float(device_locations[device_locations.sensor_id==int(sensor_id.split('_')[0])]['Latitude']), longitude=float(device_locations[device_locations.sensor_id==int(sensor_id.split('_')[0])]['Longitude']),direction=sensor_id.split('_')[1], num_neighbors_try=config_vars['num_neighbors_try'], add_heuristic=config_vars['add_heuristic'])
        list_of_devices[sensor_id] = device
    config_vars["individual_min_data_sample"] = individual_min_data_sample

    # save dataset record for resume purpose
    with open(f"{logs_dirpath}/check_point/whole_data_record.pkl", 'wb') as f:
        pickle.dump(whole_data_record, f)
        
    ''' device assign neighbors (candidate fav neighbors) '''
    for sensor_id, device in list_of_devices.items():
        device.assign_candidate_neighbors(list_of_devices)
    
    ''' device init models '''

    create_model = create_lstm if config_vars["model"] == 'lstm' else create_gru
    
    model_units = [config_vars['input_length'], config_vars['hidden_neurons'], config_vars['hidden_neurons'], config_vars['output_length']]
    model_configs = [config_vars['loss'], config_vars['optimizer'], config_vars['metric']]

    global_model_0 = create_model(model_units, model_configs)
    os.makedirs(f'{Device.logs_dirpath}/{naive_fl_global_model_path}', exist_ok=True)
    global_model_0_path = f'{Device.logs_dirpath}/{naive_fl_global_model_path}/comm_0.h5'
    global_model_0.save(global_model_0_path)
    # init models
    for sensor_id, device in list_of_devices.items():
        if config_vars['pretrained_models_path']:
            pretrained_models_folder = f'{Device.logs_dirpath}/pretrained_models'
            os.makedirs(pretrained_models_folder, exist_ok=True)
            if not os.path.exists(f"{config_vars['pretrained_models_path']}/{sensor_id}.h5"):
                print(f"WARNING - Pretrained model file for {sensor_id} NOT exists in the specified path. Init to the randomly initialized model.")
                device.init_models(f"{naive_fl_global_model_path}/comm_0.h5")
            else:
                shutil.copy(f"{config_vars['pretrained_models_path']}/{sensor_id}.h5", pretrained_models_folder)
                device.init_models(f"pretrained_models/{sensor_id}.h5")
        else:
            device.init_models(f"{naive_fl_global_model_path}/comm_0.h5")

    ''' init prediction records and fav_neighbor records'''
    device_predicts = {}
    device_TO_fav_neighbors_BY_round = {}
    for device_file in all_device_files:
        sensor_id = device_file.split('-')[-1].strip().split(".")[0]
        device_predicts[sensor_id] = {}
        # baseline 1 - central
        if int(str(config_vars["learning_methods"])[0]):
            device_predicts[sensor_id]['central'] = []
        # pure FedAvg
        if int(str(config_vars["learning_methods"])[1]):
            device_predicts[sensor_id]['naive_fl'] = []
        # radius pure FedAvg
        if int(str(config_vars["learning_methods"])[2]):
            device_predicts[sensor_id]['radius_naive_fl'] = []
        # neighbor_fl model
        if int(str(config_vars["learning_methods"])[3]):
            device_predicts[sensor_id]['neighbor_fl'] = []
        
        # ground truth
        device_predicts[sensor_id]['true'] = []
        # fav_neighbor records by comm rounds
        device_TO_fav_neighbors_BY_round[sensor_id] = []
    
    ''' get scaler '''
    scaler = get_scaler(pd.concat(list(whole_data_record.values())), config_vars['feature'])
    config_vars["scaler"] = scaler
    
    ''' save used arguments as a text file for easy review '''
    with open(f'{logs_dirpath}/args_used.txt', 'w') as f:
        f.write("Arguments used -\n")
        f.write(' '.join(sys.argv[1:]))
        f.write("\n\nAll arguments used -\n")
        for arg_name, arg in args.items():
            f.write(f'\n--{arg_name} {arg}')
        f.write("\n\nNOTE: This file records the command line and default arguments while executing the program the 1st time.")
        f.write("\nPlease refer to the args saved in config_vars.pkl for the latest used env vars.")

resume_text = f'To resume, use - \n $ python NeighborFL/main.py -rp "{logs_dirpath}"'
print(len(resume_text) * '*', "\n" + resume_text, "\n" + len(resume_text) * '*')

# init vars
get_error = vars()[f'get_{config_vars["error_type"]}']
Device.preserve_historical_models = config_vars['preserve_historical_models']

''' init FedAvg vars '''
INPUT_LENGTH = config_vars['input_length']
OUTPUT_LENGTH = config_vars['output_length']
new_sample_size_per_comm_round = config_vars['tau2p']

# determine maximum comm rounds by the minimum number of data sample a device owned and the input_length
max_comm_rounds = (individual_min_data_sample - config_vars['start_train_index'] - config_vars['tau1']) // new_sample_size_per_comm_round + 1
# comm_rounds overwritten while resuming
if config_vars['comm_rounds']:
    config_vars['comm_rounds'] = config_vars['comm_rounds']
    if max_comm_rounds > config_vars['comm_rounds']:
        print(f"\nNote: the provided dataset allows running for maximum {max_comm_rounds} comm rounds but the simulation is configured to run for {config_vars['comm_rounds']} comm rounds.")
        run_comm_rounds = config_vars['comm_rounds']
    elif config_vars['comm_rounds'] > max_comm_rounds:
        print(f"\nNote: the provided dataset ONLY allows running for maximum {max_comm_rounds} comm rounds, which is less than the configured {config_vars['comm_rounds']} comm rounds.")
        run_comm_rounds = max_comm_rounds
    else:
        run_comm_rounds = max_comm_rounds
else:
    print(f"\nNote: the provided dataset allows running for maximum {max_comm_rounds} comm rounds.")
    run_comm_rounds = max_comm_rounds

print(f"Starting Federated Learning with total comm rounds {run_comm_rounds}...")
end_data_index = config_vars["start_train_index"] + config_vars["tau1"] + (run_comm_rounds - 1) * config_vars["tau2p"]

print(f"Start training Timestamp (inclusive): {whole_data_record[list(whole_data_record.keys())[0]].iloc[config_vars['start_train_index']]['Timestamp']}")
print(f"Start prediction Timestamp (inclusive): {whole_data_record[list(whole_data_record.keys())[0]].iloc[config_vars['start_train_index'] + INPUT_LENGTH]['Timestamp']}")
print(f"End Timestamp (inclusive): {whole_data_record[list(whole_data_record.keys())[0]].iloc[end_data_index - 1]['Timestamp']}")

methods_count = str(config_vars['learning_methods']).count('1')

for comm_round in range(STARTING_COMM_ROUND, run_comm_rounds + 1): 
    print_text = f"Simulating comm_round {comm_round}/{run_comm_rounds} ({comm_round/run_comm_rounds:.0%})..."
    print("\n" + "=" * len(print_text), "\n" + print_text, "\n" + "=" * len(print_text))
    start_time = datetime.now()
    
    ''' calculate simulation data range (collect and update dataset) '''

    end_collected_data_index = config_vars['start_train_index'] + config_vars['tau1'] + (comm_round - 1) * new_sample_size_per_comm_round - 1
    
    # train data
    training_start_index = end_collected_data_index - config_vars["max_data_size"] + 1
    training_start_index = config_vars["start_train_index"] if training_start_index < config_vars["start_train_index"] else training_start_index
    # test data
    test_data_start_index = config_vars["start_train_index"] if comm_round == 1 else end_collected_data_index - new_sample_size_per_comm_round - INPUT_LENGTH - (OUTPUT_LENGTH - 1) + 1

    predict_end_indexes = [ind for ind in range(end_collected_data_index, end_collected_data_index + OUTPUT_LENGTH)]
    if comm_round == 1:
        print(f"Devices collect from data index {training_start_index} to {end_collected_data_index} (inclusive)")
        predict_start_indexes = [ind for ind in range(training_start_index + INPUT_LENGTH, training_start_index + INPUT_LENGTH + OUTPUT_LENGTH)]
        print(f"Devices predict from data index {predict_start_indexes} to {predict_end_indexes}") 
    else:
        print(f"Devices collect from data index {end_collected_data_index - config_vars['tau2p'] + 1} to {end_collected_data_index} (inclusive)")
        predict_start_indexes = [ind for ind in range(end_collected_data_index - config_vars['tau2p'] + 1, end_collected_data_index - config_vars['tau2p'] + 1 + OUTPUT_LENGTH)]
        print(f"Devices predict from data index {predict_start_indexes} to {predict_end_indexes}")
    print(f"Devices train from data index {training_start_index} to {end_collected_data_index} (inclusive)\n")

    detecotr_iter = 1
    for sensor_id, device in list_of_devices.items():

        print_text = f"Device {sensor_id}"
        print("\n" + "=" * len(print_text), "\n" + print_text, "\n" + "=" * len(print_text))

        ''' Process traning data '''
        # slice training data
        train_data = whole_data_record[sensor_id][training_start_index: end_collected_data_index + 1]
        # process training data
        X_train, y_train = process_train_data(train_data, scaler, INPUT_LENGTH, OUTPUT_LENGTH, config_vars['feature'])
        ''' Process test data '''
        # slice test data
        test_data = whole_data_record[sensor_id][test_data_start_index: end_collected_data_index + 1]
        # process test data
        X_test, y_true = process_test_data(test_data, scaler, INPUT_LENGTH, OUTPUT_LENGTH, comm_round, config_vars['feature'])
        # reshape data
        for data_set in ['X_train', 'X_test']:
            vars()[data_set] = np.reshape(vars()[data_set], (vars()[data_set].shape[0], vars()[data_set].shape[1], 1))
        y_true = scaler.inverse_transform(y_true)

        ''' Predictions '''
        device_predicts[sensor_id]['true'].append((comm_round,y_true))
        models = ['central', 'naive_fl', 'radius_naive_fl', 'neighbor_fl']
        get_models = ['central', 'last_naive_fl_global', 'last_radius_naive_fl_agg', 'last_neighbor_fl_agg']
        model_count = 1
        for i, m in enumerate(models):
            if int(str(config_vars["learning_methods"])[i]):
                print_text = f"{sensor_id} ({detecotr_iter}/{len(list_of_devices)}) predicting by {m} ({model_count}/{methods_count})..."
                print("\n" + '-' * len(print_text) + f'\n{print_text}')
                model = getattr(device, f"get_{get_models[i]}_model")()
                predictions = model.predict(X_test)
                predictions = scaler.inverse_transform(predictions)
                device_predicts[sensor_id][m].append((comm_round, predictions))
                model_count += 1 

        ''' Training, except NeighborFL '''
        models = ['central', 'naive_fl', 'radius_naive_fl']
        get_models = ['central', 'last_naive_fl_global', 'last_radius_naive_fl_agg']
        save_model_paths = ['central', 'naive_fl_local', 'radius_naive_fl_local']
        model_count = 1
        for i, m in enumerate(models):
            if int(str(config_vars["learning_methods"])[i]):
                print_text = f"{sensor_id} ({detecotr_iter}/{len(list_of_devices)}) training {m} ({model_count}/{methods_count})..."
                print("\n" + '-' * len(print_text) + f'\n{print_text}')
                model = getattr(device, f"get_{get_models[i]}_model")()
                new_model = train_model(model, X_train, y_train, config_vars['batch'], config_vars['epochs'])
                device.update_and_save_model(new_model, comm_round, vars()[f'{save_model_paths[i]}_model_path'])
                model_count += 1
        
        ''' NeighborFL - Evaluating candidate, training'''
        if int(str(config_vars["learning_methods"])[3]):
            # neighbor_fl local model
            device.has_added_neigbor = False
            chosen_model = device.get_last_neighbor_fl_agg_model() # by default
            
            # for O > 1, special dealing with y_true and predicts
            predicts = device_predicts[sensor_id]['neighbor_fl'][-1][1][:-(OUTPUT_LENGTH - 1)] # drop latest
            if comm_round > 1:
                y_true = y_true[OUTPUT_LENGTH - 1:] # drop earlist
            error_without_new_neighbors = get_error(y_true, predicts)  # also works when not selected candidates in last round
            
            neighbor_fl_error = error_without_new_neighbors
            device.neighbor_fl_error_records.append(neighbor_fl_error)
                
            if device.eval_neighbors:
                print(f"{sensor_id} comparing models with and without candidate neighbor(s) to determine which model to train...")
                # do prediction

                print(f"{sensor_id} predicting by the eval_neighbor_fl_agg_model (has the model from the newly eval neighbor(s)).")
                eval_neighbor_fl_agg_model = device.get_eval_neighbor_fl_agg_model()
                eval_neighbor_fl_agg_model_predictions = eval_neighbor_fl_agg_model.predict(X_test)
                eval_neighbor_fl_agg_model_predictions = scaler.inverse_transform(eval_neighbor_fl_agg_model_predictions)
                
                # drop latest
                eval_predicts = eval_neighbor_fl_agg_model_predictions[:-(OUTPUT_LENGTH - 1)]
                error_with_new_neighbors = get_error(y_true, eval_predicts)
                error_diff = error_without_new_neighbors - error_with_new_neighbors
                if error_diff > 0:
                    # candidate neighbor(s) are good
                    print(f"eval_neighbor_fl_agg_model will be used for local model update.")
                    chosen_model = device.get_eval_neighbor_fl_agg_model()
                for neighbor in device.eval_neighbors:
                    if error_diff > 0:
                        # candidate neighbor(s) are good
                        device.fav_neighbors.append(neighbor)
                        device.has_added_neigbor = True
                        print(f"{device.id} added {neighbor.id} to its fav neighbors!")
                        # update neighbor_to_accumulate_interval
                        if neighbor.id in device.neighbor_to_accumulate_interval and device.neighbor_to_accumulate_interval[neighbor.id] > 0:
                            device.neighbor_to_accumulate_interval[neighbor.id] -= 1
                    else:
                        # candidate neighbor(s) are bad
                        print(f"{device.id} SKIPPED adding {neighbor.id} to its fav neighbors.")
                        device.neighbor_to_last_accumulate[neighbor.id] = comm_round
                        device.neighbor_to_accumulate_interval[neighbor.id] = device.neighbor_to_accumulate_interval.get(neighbor.id, 0) + 1
                    # give reputation
                    # 3/12/23, since traffic always dynamically changes, newer round depends on older rounds error may not be reliable
                    device.neighbors_to_rep_score[neighbor.id] = device.neighbors_to_rep_score.get(neighbor.id, 0) + error_diff
            
            print_text = f"{sensor_id} training neighbor_fl local model.. ({model_count}/{methods_count})"
            print("\n" + '-' * len(print_text) + f'\n{print_text}')
            new_model = train_model(chosen_model, X_train, y_train, config_vars['batch'], config_vars['epochs'])
            device.update_and_save_model(new_model, comm_round, neighbor_fl_local_model_path)
            # record current comm_round of fav_neighbors for visualization
            current_fav_neighbors = set(fav_neighbor.id for fav_neighbor in device.fav_neighbors)
            device_TO_fav_neighbors_BY_round[sensor_id].append(current_fav_neighbors)
            print(f"{sensor_id}'s current fav neighbors are {current_fav_neighbors}.")
            model_count += 1

        detecotr_iter += 1
    
    ''' Simulate Global/Aggregated Model Aggregation'''

    model_count = 1
    if int(str(config_vars["learning_methods"])[1]):
        ''' Simulate NaiveFL FedAvg '''
        print_text = f"All devices simulating NaiveFL aggregation ({model_count}/{methods_count})"
        print('\n', '-' * len(print_text) + f'\n{print_text}')
        # create naive_fl model from all naive_fl_local models
        new_naive_fl_global_model = create_model(model_units, model_configs)
        naive_fl_local_models_weights = []
        for sensor_id, device in list_of_devices.items():
            naive_fl_local_models_weights.append(device.get_naive_fl_local_model().get_weights())
        new_naive_fl_global_model.set_weights(np.mean(naive_fl_local_models_weights, axis=0))
        # save new_naive_fl_global_model
        Device.save_fl_global_model(new_naive_fl_global_model, comm_round, naive_fl_global_model_path)
    
        detecotr_iter = 1
        for sensor_id, device in list_of_devices.items():
            # update new_naive_fl_global_model
            device.update_naive_fl_global_model(comm_round, naive_fl_global_model_path)
            detecotr_iter += 1
        model_count += 1

    if int(str(config_vars["learning_methods"])[2]):
        ''' Simulate r-NaiveFL FedAvg '''
        detecotr_iter = 1
        for sensor_id, device in list_of_devices.items():
            print_text = f"{sensor_id} ({detecotr_iter}/{len(list_of_devices)}) simulating r-NaiveFL aggregation ({model_count}/{methods_count})"
            print('\n', '-' * len(print_text) + f'\n{print_text}')
            
            # create radius_naive_fl_agg_model based on the current fav neighbors
            radius_naive_fl_agg_model = create_model(model_units, model_configs)
            radius_naive_fl_agg_models_weights = {}
            radius_naive_fl_agg_models_weights[device.id] = device.get_radius_naive_fl_local_model().get_weights()
            for radius_neighbor, _ in device.candidate_neighbors:
                radius_naive_fl_agg_models_weights[radius_neighbor.id] = radius_neighbor.get_radius_naive_fl_local_model().get_weights()
            radius_naive_fl_agg_model.set_weights(np.mean(list(radius_naive_fl_agg_models_weights.values()), axis=0))
            # save model
            device.update_and_save_model(radius_naive_fl_agg_model, comm_round, radius_naive_fl_agg_model_path)
            detecotr_iter += 1
        model_count += 1
    
    if int(str(config_vars["learning_methods"])[3]):
        ''' Simulate NeighborFL FedAvg '''    
        # determine if add new neighbor or not (core algorithm)
        detecotr_iter = 1
        for sensor_id, device in list_of_devices.items():
            print_text = f"{sensor_id} ({detecotr_iter}/{len(list_of_devices)}) simulating NeighborFL ({model_count}/{methods_count})"
            print('\n', '-' * len(print_text) + f'\n{print_text}')
            # create neighbor_fl_agg_model based on the current fav neighbors
            neighbor_fl_agg_model = create_model(model_units, model_configs)
            neighbor_fl_agg_models_weights = {}
            neighbor_fl_agg_models_weights[device.id] = device.get_neighbor_fl_local_model().get_weights()
            for fav_neighbor in device.fav_neighbors:
                neighbor_fl_agg_models_weights[fav_neighbor.id] = fav_neighbor.get_neighbor_fl_local_model().get_weights()
            neighbor_fl_agg_model.set_weights(np.mean(list(neighbor_fl_agg_models_weights.values()), axis=0))
            # save model
            device.update_and_save_model(neighbor_fl_agg_model, comm_round, neighbor_fl_agg_model_path)
            
            ''' remove some fav neighbors by rolling a dice and strategy '''
            if_remove = False
            if config_vars["remove_trigger"] == 1 and random.random() <= config_vars['epsilon']:
                if_remove = True
            if config_vars["remove_trigger"] == 2 and len(device.neighbor_fl_error_records) > config_vars['remove_rounds'] and not device.has_added_neigbor:
                last_rounds_error = device.neighbor_fl_error_records[-(config_vars['remove_rounds'] + 1):]
                if all(x<y for x, y in zip(last_rounds_error, last_rounds_error[1:])):
                    if_remove = True
            # remove
            if if_remove:
                remove_num = config_vars["remove_num"]
                rep_tuples = [(id, rep) for id, rep in sorted(device.neighbors_to_rep_score.items(), key=lambda x: x[1])]
                if config_vars["remove_strategy"] == 1:
                    # remove by lowest reputation
                    # May need improve - since traffic always dynamically changes, newer round depends on older rounds error may not be reliable
                    for i in range(len(rep_tuples)):
                        if remove_num > 0:
                            to_remove_id = rep_tuples[i][0]
                            if list_of_devices[to_remove_id] in device.fav_neighbors:
                                device.fav_neighbors.remove(list_of_devices[to_remove_id])
                                print(f"{sensor_id} removes out {to_remove_id}, leaving {set(fav_neighbor.id for fav_neighbor in device.fav_neighbors)}.")
                                remove_num -= 1
                                # add retry interval since experiment shows that later in try phase, the same neighbor may be eval again
                                device.neighbor_to_last_accumulate[to_remove_id] = comm_round
                                device.neighbor_to_accumulate_interval[to_remove_id] = device.neighbor_to_accumulate_interval.get(to_remove_id, 0) + 1
                        else:
                            break
                elif config_vars["remove_strategy"] == 2:
                    # remove randomly
                    for i in range(len(rep_tuples)):
                        if remove_num > 0:
                            removeed_neighbor = device.fav_neighbors.pop(random.randrange(len(device.fav_neighbors)))
                            print(f"{sensor_id} removes out {removeed_neighbor.id}, leaving {set(fav_neighbor.id for fav_neighbor in device.fav_neighbors)}.")
                            remove_num -= 1
                            device.neighbor_to_last_accumulate[removeed_neighbor.id] = comm_round
                            device.neighbor_to_accumulate_interval[removeed_neighbor.id] = device.neighbor_to_accumulate_interval.get(removeed_neighbor.id, 0) + 1
                        else:
                            break
                elif config_vars["remove_strategy"] == 3:
                    # always remove the last added one
                    if len(device.fav_neighbors) > 0:
                        removeed = device.fav_neighbors.pop()
                        print(f"{sensor_id} removes out {removeed.id}, leaving {set(fav_neighbor.id for fav_neighbor in device.fav_neighbors)}.")
                        del neighbor_fl_agg_models_weights[removeed.id]
                        device.neighbor_to_last_accumulate[removeed.id] = comm_round
                        device.neighbor_to_accumulate_interval[removeed.id] = device.neighbor_to_accumulate_interval.get(removeed.id, 0) + 1
                


            ''' evaluate new neighbor(s)? '''
            try_new_neighbors = False
            if len(device.fav_neighbors) < len(device.candidate_neighbors):
                try_new_neighbors = True
            else:
                try_new_neighbors = True
            
            eval_neighbor_fl_agg_model = create_model(model_units, model_configs)
            device.eval_neighbors = []
            candidate_count = min(config_vars['num_neighbors_try'], len(device.candidate_neighbors) - len(device.fav_neighbors))
            candidate_iter = 0
            while candidate_count > 0 and try_new_neighbors and candidate_iter < len(device.candidate_neighbors):
                candidate_fav = device.candidate_neighbors[candidate_iter][0]
                if candidate_fav not in device.fav_neighbors:
                    if candidate_fav.id in device.neighbor_to_last_accumulate:
                        if comm_round <= device.neighbor_to_last_accumulate[candidate_fav.id] + device.neighbor_to_accumulate_interval[candidate_fav.id]:
                            # skip this device
                            candidate_iter += 1
                            continue
                    device.eval_neighbors.append(candidate_fav)
                    print(f"{sensor_id} selects {candidate_fav.id} to evaluate next round.")
                    neighbor_fl_agg_models_weights[candidate_fav.id] = candidate_fav.get_neighbor_fl_local_model().get_weights()
                    candidate_count -= 1
                candidate_iter += 1
                        
            eval_neighbor_fl_agg_model.set_weights(np.mean(list(neighbor_fl_agg_models_weights.values()), axis=0))
            # Note - when there is no eval neighbor, a eval_neighbor_fl_agg_model will also be saved (and identical to neighbor_fl_agg_model), but won't used in the next round
            device.update_and_save_model(eval_neighbor_fl_agg_model, comm_round, eval_neighbor_fl_agg_model_path)
            
            # if heuristic is randomly choosing candidate neighbors, reshuffle
            if config_vars["add_heuristic"] == 2:
                device.candidate_neighbors = random.shuffle(device.candidate_neighbors)
                    
            detecotr_iter += 1
        model_count += 1

    end_time = datetime.now()
    print_text = f"Comm round {comm_round} takes {(end_time-start_time).total_seconds() / 60} minutes."
    print("\n" + "-" * len(print_text) + "\n" + print_text)
    
    print(f"Saving progress for comm_round {comm_round}...")

    print("Saving Predictions...")                          
    predictions_record_saved_path = f'{logs_dirpath}/check_point/all_device_predicts.pkl'
    with open(predictions_record_saved_path, 'wb') as f:
        pickle.dump(device_predicts, f)

    print("Saving Fav Neighbors of All Detecors...")
    fav_neighbors_record_saved_path = f'{logs_dirpath}/check_point/fav_neighbors_by_round.pkl'
    with open(fav_neighbors_record_saved_path, 'wb') as f:
        pickle.dump(device_TO_fav_neighbors_BY_round, f)
    
    print("Saving Resume Params...")
    config_vars["resume_comm_round"] = comm_round + 1
    with open(f"{logs_dirpath}/check_point/config_vars.pkl", 'wb') as f:
        pickle.dump(config_vars, f)
    with open(f"{logs_dirpath}/check_point/list_of_devices.pkl", 'wb') as f:
        pickle.dump(list_of_devices, f)

print(f"Simulation done for {run_comm_rounds} comm rounds.")