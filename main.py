
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

from build_lstm import build_lstm
from build_gru import build_gru
from model_training import train_model

from process_data import get_scaler
from process_data import process_train_data
from process_data import process_test_one_step
from process_data import process_test_multi_and_get_y_true

from error_calc import get_MAE
from error_calc import get_MSE
from error_calc import get_RMSE
from error_calc import get_MAPE

import random

# remove some warnings
import logging
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

''' Parse command line arguments '''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="traffic_fedavg_simulation")

# arguments for system vars
parser.add_argument('-dp', '--dataset_path', type=str, default='/content/drive/MyDrive/traffic_data/', help='dataset path')
parser.add_argument('-lb', '--logs_base_folder', type=str, default="/content/drive/MyDrive/KFRT_logs", help='base folder path to store running logs and h5 files')
parser.add_argument('-pm', '--preserve_historical_models', type=int, default=0, help='whether to preserve models from old communication comm_rounds. Consume storage. Input 1 to preserve')

# arguments for resume training
parser.add_argument('-rp', '--resume_path', type=str, default=None, help='provide the leftover log folder path to continue FL')

# arguments for learning
parser.add_argument('-m', '--model', type=str, default='lstm', help='Model to choose - lstm or gru')
parser.add_argument('-il', '--input_length', type=int, default=12, help='input length for the LSTM/GRU network')
parser.add_argument('-hn', '--hidden_neurons', type=int, default=128, help='number of neurons in one of each 2 layers')
parser.add_argument('-b', '--batch', type=int, default=1, help='batch number for training')
parser.add_argument('-e', '--epochs', type=int, default=5, help='epoch number per comm comm_round for FL')
parser.add_argument('-ff', '--num_feedforward', type=int, default=12, help='number of feedforward predictions, used to set up the number of the last layer of the model (usually it has to be equal to -il)')
parser.add_argument('-tp', '--train_percent', type=float, default=1.0, help='percentage of the data for training')

# arguments for federated learning
parser.add_argument('-c', '--comm_rounds', type=int, default=None, help='number of comm rounds, default aims to run until data is exhausted')
parser.add_argument('-ms', '--max_data_size', type=int, default=24, help='maximum data length for training in each communication comm_round, simulating the memory space a detector has')

# arguments for fav_neighbor fl
parser.add_argument('-r', '--radius', type=float, default=None, help='only treat the participants within radius as neighbors')
parser.add_argument('-k', '--k', type=float, default=None, help='maximum number of fav_neighbors. radius and k can be used together, but radius has the priority')
parser.add_argument('-et', '--error_type', type=str, default="MAE", help='the error type to evaluate potential neighbors')
parser.add_argument('-nt', '--num_neighbors_try', type=int, default=1, help='how many new neighbors to try in each comm_round')
parser.add_argument('-ep', '--epsilon', type=float, default=0.2, help='detector has a probability to kick out worst neighbors to explore new neighbors')
parser.add_argument('-kn', '--kick_num', type=int, default=2, help='two strategies: 1 - kick 1 worst detector, 2 - kick 25% of the worst detectors')
parser.add_argument('-ks', '--kick_strategy', type=int, default=1, help='0 - never kick, 1 - kick by worst reputation, 2 - kick randomly')
parser.add_argument('-ah', '--add_heuristic', type=int, default="1", help='heuristic to add fav neighbors: 1 - add by distance from close to far, 2 - add randomly')

args = parser.parse_args()
args = args.__dict__
''' Parse command line arguments '''

stand_alone_model_path = 'stand_alone'
naive_fl_local_model_path = 'naive_fl_local'
naive_fl_global_model_path = 'naive_fl_global'
fav_neighbors_fl_local_model_path = 'fav_neighbors_fl_local'
fav_neighbors_fl_agg_model_path = 'fav_neighbors_fl_agg'

print("Preparing - i.e., create detector objects, init models, load data, etc,.\nThis may take a few minutes...")
# determine if resume training
if args['resume_path']:
    logs_dirpath = args['resume_path']
    Detector.logs_dirpath = logs_dirpath
    # load saved variables
    with open(f"{logs_dirpath}/config_vars.pkl", 'rb') as f:
        # overwrite all args
        config_vars = pickle.load(f)
        config_vars['resume_path'] = logs_dirpath
    with open(f'{logs_dirpath}/all_detector_predicts.pkl', 'rb') as f:
        detector_predicts = pickle.load(f)
    with open(f'{logs_dirpath}/fav_neighbors.pkl', 'rb') as f:
        detector_fav_neighbors = pickle.load(f)
    with open(f"{logs_dirpath}/list_of_detectors.pkl", 'rb') as f:
        list_of_detectors = pickle.load(f)
    with open(f"{logs_dirpath}/whole_data_record.pkl", 'rb') as f:
        whole_data_record = pickle.load(f)
    STARTING_COMM_ROUND = config_vars["resume_comm_round"]
    scaler = config_vars["scaler"]
    individual_min_data_sample = config_vars["individual_min_data_sample"]
    detector_locations = config_vars["detector_locations"]
    
    build_model = build_lstm if config_vars["model"] == 'lstm' else build_gru
else:
    ''' logistics '''
    # save command line arguments
    config_vars = args
    STARTING_COMM_ROUND = 1
    
    # read in detector file paths
    all_detector_files = [f for f in listdir(args["dataset_path"]) if isfile(join(args["dataset_path"], f)) and '.csv' in f and not 'location' in f]
    print(f'We have {len(all_detector_files)} detectors available.')
    # read in detector location file
    detector_location_file = [f for f in listdir(args["dataset_path"]) if isfile(join(args["dataset_path"], f)) and 'location' in f][0]
    detector_locations = pd.read_csv(os.path.join(args['dataset_path'], detector_location_file))
    
    config_vars["detector_locations"] = detector_locations

    # create log folder indicating by current running date and time
    date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    logs_dirpath = f"{args['logs_base_folder']}/{date_time}_{args['model']}_input_{args['input_length']}_mds_{args['max_data_size']}_epoch_{args['epochs']}"
    os.makedirs(logs_dirpath, exist_ok=True)
    Detector.logs_dirpath = logs_dirpath
    
    ''' create detector object and load data for each detector '''
    whole_data_record = {} # to calculate scaler
    individual_min_data_sample = float('inf') # to determine max comm rounds
    list_of_detectors = {}
    for detector_file_iter in range(len(all_detector_files)):
        detector_file_name = all_detector_files[detector_file_iter]
        detector_id = detector_file_name.split('.')[0]
        # data file path
        file_path = os.path.join(config_vars['dataset_path'], detector_file_name)
        
        # count lines to later determine max comm rounds
        whole_data = pd.read_csv(file_path, encoding='utf-8').fillna(0)
        read_to_line = int(whole_data.shape[0] * config_vars["train_percent"])
        individual_min_data_sample = read_to_line if read_to_line < individual_min_data_sample else individual_min_data_sample
        
        whole_data = whole_data[:read_to_line]
        print(f'Loaded {read_to_line} lines of data from {detector_file_name} (percentage: {config_vars["train_percent"]}). ({detector_file_iter+1}/{len(list_of_detectors.keys())})')
        whole_data_record[detector_id] = whole_data
        # create a detector object
        detector = Detector(detector_id, radius=config_vars['radius'], k=config_vars['k'],latitude=float(detector_locations[detector_locations.device_id==int(detector_id.split('_')[0])]['lat']), longitude=float(detector_locations[detector_locations.device_id==int(detector_id.split('_')[0])]['lon']),direction=detector_id.split('_')[1], num_neighbors_try=config_vars['num_neighbors_try'], add_heuristic=config_vars['add_heuristic'], epsilon=config_vars['epsilon'])
        list_of_detectors[detector_id] = detector
    config_vars["individual_min_data_sample"] = individual_min_data_sample
    
    # save dataset record for resume purpose
    with open(f"{logs_dirpath}/whole_data_record.pkl", 'wb') as f:
        pickle.dump(whole_data_record, f)
        
    ''' detector assign neighbors (candidate fav neighbors) '''
    for detector_id, detector in list_of_detectors.items():
        detector.assign_neighbors(list_of_detectors)
    
    ''' detector init models '''

    build_model = build_lstm if config_vars["model"] == 'lstm' else build_gru

    global_model_0 = build_model([config_vars['input_length'], config_vars['hidden_neurons'], config_vars['hidden_neurons'], 1])
    global_model_0.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    os.makedirs(f'{Detector.logs_dirpath}/{naive_fl_global_model_path}', exist_ok=True)
    global_model_0_path = f'{Detector.logs_dirpath}/{naive_fl_global_model_path}/comm_0.h5'
    global_model_0.save(global_model_0_path)
    # init models
    for detector_id, detector in list_of_detectors.items():
        detector.init_models(f"{naive_fl_global_model_path}/comm_0.h5")
        
    ''' init prediction records and fav_neighbor records'''
    detector_predicts = {}
    detector_fav_neighbors = {}
    for detector_file in all_detector_files:
        detector_id = detector_file.split('.')[0]
        detector_predicts[detector_id] = {}
        # baseline 1 - stand_alone
        detector_predicts[detector_id]['stand_alone'] = []
        # baseline 2 - naive global
        detector_predicts[detector_id]['naive_fl'] = []
        # fav_neighbors_fl model
        detector_predicts[detector_id]['fav_neighbors_fl'] = []
        # true
        detector_predicts[detector_id]['true'] = []
        # fav_neighbor records
        detector_fav_neighbors[detector_id] = []
    
    ''' get scaler '''
    scaler = get_scaler(pd.concat(list(whole_data_record.values())))
    config_vars["scaler"] = scaler
    
    ''' save used arguments as a text file for easy review '''
    with open(f'{logs_dirpath}/args_used.txt', 'w') as f:
        f.write("Command line arguments used -\n")
        f.write(' '.join(sys.argv[1:]))
        f.write("\n\nAll arguments used -\n")
        for arg_name, arg in args.items():
            f.write(f'\n--{arg_name} {arg}')

# init vars
get_error = vars()[f'get_{config_vars["error_type"]}']
Detector.preserve_historical_models = config_vars['preserve_historical_models']

''' init FedAvg vars '''
INPUT_LENGTH = config_vars['input_length']
new_sample_size_per_comm_round = INPUT_LENGTH

# determine maximum comm rounds by the minimum number of data sample a device owned and the input_length
max_comm_rounds = individual_min_data_sample // INPUT_LENGTH - 2
# comm_rounds overwritten while resuming
if args['comm_rounds']:
    config_vars['comm_rounds'] = args['comm_rounds']
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

for comm_round in range(STARTING_COMM_ROUND, run_comm_rounds + 1):
    print(f"Simulating comm comm_round {comm_round}/{run_comm_rounds} ({comm_round/run_comm_rounds:.0%})...")
    ''' calculate simulation data range '''
    # train data
    if comm_round == 1:
        training_data_starting_index = 0
        training_data_ending_index = training_data_starting_index + new_sample_size_per_comm_round * 2 - 1
        # if it's comm_round 1 and input_shape 12, need at least 24 training data points because the model at least needs 13 points to train.
        # Therefore,
        # comm_round 1 -> 1~24 training points, predict with test 13~35 test points, 
        # 1- 24， 2 - 36， 3 - 48， 4 - 60
    else:
        training_data_ending_index = (comm_round + 1) * new_sample_size_per_comm_round - 1
        training_data_starting_index = training_data_ending_index - config_vars['max_data_size']
        if training_data_starting_index < 1:
            training_data_starting_index = 0
    # test data
    test_data_starting_index = training_data_ending_index - new_sample_size_per_comm_round + 1
    test_data_ending_index_one_step = test_data_starting_index + new_sample_size_per_comm_round * 2 - 1
    test_data_ending_index_chained_multi = test_data_starting_index + new_sample_size_per_comm_round - 1
    
    detecotr_iter = 1
    for detector_id, detector in list_of_detectors.items():
        ''' Process traning data '''
        # slice training data
        train_data = whole_data_record[detector_id][training_data_starting_index: training_data_ending_index + 1]
        # process training data
        X_train, y_train = process_train_data(train_data, scaler, INPUT_LENGTH)
        ''' Process test data '''
        # slice test data
        test_data = whole_data_record[detector_id][test_data_starting_index: test_data_ending_index_one_step + 1]
        # process test data
        X_test, _ = process_test_one_step(test_data, scaler, INPUT_LENGTH)
        _, y_true = process_test_multi_and_get_y_true(test_data, scaler, INPUT_LENGTH, config_vars['num_feedforward'])
        # reshape data
        for data_set in ['X_train', 'X_test']:
            vars()[data_set] = np.reshape(vars()[data_set], (vars()[data_set].shape[0], vars()[data_set].shape[1], 1))
        y_true = scaler.inverse_transform(y_true.reshape(-1, 1)).reshape(1, -1)[0]
        # record data
        detector.set_X_test(X_test)
        detector.set_y_true(y_true)
        detector_predicts[detector_id]['true'].append((comm_round,y_true))
        
        ''' Training '''
        print(f"{detector_id} ({detecotr_iter}/{len(list_of_detectors.keys())}) now training on row {training_data_starting_index} to {training_data_ending_index}...")
        # stand_alone model
        print(f"{detector_id} training stand_alone model.. (1/3)")
        new_model = train_model(detector.get_stand_alone_model(), X_train, y_train, config_vars['batch'], config_vars['epochs'])
        detector.update_and_save_stand_alone_model(new_model, comm_round, stand_alone_model_path)
        # naive_fl local model
        print(f"{detector_id} training naive_fl local model.. (2/3)")
        new_model = train_model(detector.get_last_naive_fl_global_model(), X_train, y_train, config_vars['batch'], config_vars['epochs'])
        detector.update_and_save_naive_fl_local_model(new_model, comm_round, naive_fl_local_model_path)
        # fav_neighbors_fl local model
        print(f"{detector_id} training fav_neighbors_fl local model.. (3/3)")
        new_model = train_model(detector.get_last_fav_neighbors_fl_agg_model(), X_train, y_train, config_vars['batch'], config_vars['epochs'])
        detector.update_and_save_fav_neighbors_fl_local_model(new_model, comm_round, fav_neighbors_fl_local_model_path)
        
        ''' stand_alone model predictions '''
        print(f"{detector_id} is now predicting by its stand_alone model...")
        stand_alone_predictions = detector.get_stand_alone_model().predict(X_test)
        stand_alone_predictions = scaler.inverse_transform(stand_alone_predictions.reshape(-1, 1)).reshape(1, -1)[0]
        detector_predicts[detector_id]['stand_alone'].append((comm_round,stand_alone_predictions))

        detecotr_iter += 1
        
    ''' Simulate Vanilla CCGrid FedAvg '''
    # create naive_fl model from all naive_fl_local models
    print("Predicting on naive fl model")
    new_naive_fl_global_model = build_model([config_vars['input_length'], config_vars['hidden_neurons'], config_vars['hidden_neurons'], 1])
    new_naive_fl_global_model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    naive_fl_local_models_weights = []
    for detector_id, detector in list_of_detectors.items():
        naive_fl_local_models_weights.append(detector.get_naive_fl_local_model().get_weights())
    new_naive_fl_global_model.set_weights(np.mean(naive_fl_local_models_weights, axis=0))
    # save new_naive_fl_global_model
    Detector.save_fl_global_model(new_naive_fl_global_model, comm_round, naive_fl_global_model_path)
    
    detecotr_iter = 1
    for detector_id, detector in list_of_detectors.items():
        # update new_naive_fl_global_model
        detector.update_fl_global_model(comm_round, naive_fl_global_model_path)
        # do prediction
        print(f"{detector_id} ({detecotr_iter}/{len(list_of_detectors.keys())}) now predicting by new_naive_fl_global_model")
        new_naive_fl_global_model_predictions = new_naive_fl_global_model.predict(detector.get_X_test())
        new_naive_fl_global_model_predictions = scaler.inverse_transform(new_naive_fl_global_model_predictions.reshape(-1, 1)).reshape(1, -1)[0]
        detector
        detector_predicts[detector_id]['naive_fl'].append((comm_round,new_naive_fl_global_model_predictions))
        detecotr_iter += 1
    
    ''' Simulate fav_neighbor FL FedAvg '''    
    # determine if add new neighbor or not (core algorithm)
    detecotr_iter = 1
    for detector_id, detector in list_of_detectors.items():
        ''' evaluate new potential neighbors' models '''
        print_text = f"{detector_id} ({detecotr_iter}/{len(list_of_detectors.keys())}) simulating fav_neighbor FL"
        print('-' * len(print_text))
        print(print_text)
        if detector.tried_neighbors:
            print(f"{detector_id} now evaluating new potential fav neighbors' ({set(d.id for d in detector.tried_neighbors)}) models.")
            error_without_new_neighbors = get_error(y_true, detector.fav_neighbors_fl_predictions)
            error_with_new_neighbors = get_error(y_true, detector.to_compare_fav_neighbors_fl_predictions)
            error_diff = error_without_new_neighbors - error_with_new_neighbors
            for neighbor in detector.tried_neighbors:
                if error_diff > 0:
                    # tried neighbors are good
                    detector.fav_neighbors.append(neighbor)
                    print(f"{neighbor.id} added to {detector.id}'s fav neighbors!")
                else:
                    # tried neighbors are bad
                    print(f"{detector.id} SKIPPED adding {neighbor.id} to its fav neighbors.")
                    detector.neighbor_to_last_accumulate[neighbor.id] = comm_round - 1
                    detector.neighbor_to_accumulate_interval[neighbor.id] = detector.neighbor_to_accumulate_interval.get(neighbor.id, 0) + 1
                # give reputation
                detector.neighbors_to_rep_score[neighbor.id] = detector.neighbor_to_accumulate_interval.get(neighbor.id, 0) + error_diff
        # record current comm_round of fav_neighbors for visualization
        current_fav_neighbors = set(fav_neighbor.id for fav_neighbor in detector.fav_neighbors)
        detector_fav_neighbors[detector_id].append(current_fav_neighbors)
        print(f"{detector_id}'s current fav neighbors are {current_fav_neighbors}.")
        # create fav_neighbors_fl_agg_model based on the current fav neighbors
        fav_neighbors_fl_agg_model = build_model([config_vars['input_length'], config_vars['hidden_neurons'], config_vars['hidden_neurons'], 1])
        fav_neighbors_fl_agg_model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
        fav_neighbors_fl_agg_models_weights = [detector.get_fav_neighbors_fl_local_model().get_weights()]
        for fav_neighbor in detector.fav_neighbors:
            fav_neighbors_fl_agg_models_weights.append(fav_neighbor.get_fav_neighbors_fl_local_model().get_weights())
        fav_neighbors_fl_agg_model.set_weights(np.mean(fav_neighbors_fl_agg_models_weights, axis=0))
        # save model
        detector.update_and_save_fav_neighbors_fl_agg_model(fav_neighbors_fl_agg_model, comm_round, fav_neighbors_fl_agg_model_path)
        # do prediction
        print(f"{detector_id} now predicting by its fav_neighbors_fl_agg_model.")
        fav_neighbors_fl_agg_model_predictions = fav_neighbors_fl_agg_model.predict(detector.get_X_test())
        fav_neighbors_fl_agg_model_predictions = scaler.inverse_transform(fav_neighbors_fl_agg_model_predictions.reshape(-1, 1)).reshape(1, -1)[0]
        detector_predicts[detector_id]['fav_neighbors_fl'].append((comm_round,fav_neighbors_fl_agg_model_predictions))
        detector.fav_neighbors_fl_predictions = fav_neighbors_fl_agg_model_predictions
        
        ''' if len(fav_neighbors) < k, try new neighbors! '''
        try_new_neighbors = False
        if detector.k:
            if len(detector.fav_neighbors) < detector.k:
                try_new_neighbors = True
        else:
            try_new_neighbors = True
        
        # create temporary
        temp_model = build_model([config_vars['input_length'], config_vars['hidden_neurons'], config_vars['hidden_neurons'], 1])
        temp_model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
        detector.tried_neighbors = []
        candidate_count = min(config_vars['num_neighbors_try'], len(detector.neighbors) - len(detector.fav_neighbors))
        candidate_iter = 0
        while candidate_count > 0 and try_new_neighbors and candidate_iter < len(detector.neighbors):
            # k may be set to a very large number, so checking candidate_count is still necessary
            candidate_fav = detector.neighbors[candidate_iter][0]
            if candidate_fav not in detector.fav_neighbors:
                if candidate_fav.id in detector.neighbor_to_last_accumulate:
                    if comm_round <= detector.neighbor_to_last_accumulate[candidate_fav.id] + detector.neighbor_to_accumulate_interval[candidate_fav.id]:
                        # skip this detector
                        candidate_iter += 1
                        continue
                detector.tried_neighbors.append(candidate_fav)
                print(f"{detector_id} selects {candidate_fav.id} as a new potential neighbor.")
                fav_neighbors_fl_agg_models_weights.append(candidate_fav.get_fav_neighbors_fl_local_model().get_weights())
                candidate_count -= 1
            candidate_iter += 1
                    
        temp_model.set_weights(np.mean(fav_neighbors_fl_agg_models_weights, axis=0))
        # do prediction
        print(f"{detector_id} now predicting by the fav_neighbors_fl_agg_model with updated tried neighbors.")
        temp_model_predictions = temp_model.predict(detector.get_X_test())
        temp_model_predictions = scaler.inverse_transform(temp_model_predictions.reshape(-1, 1)).reshape(1, -1)[0]
        detector.to_compare_fav_neighbors_fl_predictions = temp_model_predictions
        
        # if heuristic is randomly choosing candidate neighbors, reshuffle
        if config_vars["add_heuristic"] == 2:
            detector.neighbors = random.shuffle(detector.neighbors)
        
        ''' kick some fav neighbors by rolling a dice and strategy '''
        if config_vars["kick_strategy"]:
            if random.random() <= detector.epsilon:
                kick_num = 1
                if config_vars["kick_num"] == 2:
                    kick_num = round(len(detector.fav_neighbors) * 0.25)
                # kick 
                if config_vars["kick_strategy"] == 1:
                    # kick by lowest reputation
                    rep_tuples = [(id, rep) for id, rep in sorted(detector.neighbors_to_rep_score.items(), key=lambda x: x[1])]
                    for i in range(kick_num):
                        if list_of_detectors[rep_tuples[i][0]] in detector.fav_neighbors:
                            detector.fav_neighbors.remove(list_of_detectors[id])
                            print(f"{detector_id} kicks out {id}")
                else:
                    # kick randomly
                    for i in range(kick_num):
                        kicked_neighbor = detector.fav_neighbors.pop(random.randrange(len(detector.fav_neighbors)))
                        print(f"{detector_id} kicks out {kicked_neighbor.id}")
        detecotr_iter += 1
    
    print(f"Saving progress for comm_round {comm_round}...")
    
    print("Saving Predictions...")                          
    predictions_record_saved_path = f'{logs_dirpath}/all_detector_predicts.pkl'
    with open(predictions_record_saved_path, 'wb') as f:
        pickle.dump(detector_predicts, f)

    print("Saving Fav Neighbors of All Detecors...")
    fav_neighbors_record_saved_path = f'{logs_dirpath}/fav_neighbors.pkl'
    with open(fav_neighbors_record_saved_path, 'wb') as f:
        pickle.dump(detector_fav_neighbors, f)
    
    print("Saving Resume Params...")
    config_vars["resume_comm_round"] = comm_round + 1
    with open(f"{logs_dirpath}/config_vars.pkl", 'wb') as f:
        pickle.dump(config_vars, f)
    with open(f"{logs_dirpath}/list_of_detectors.pkl", 'wb') as f:
        pickle.dump(list_of_detectors, f)


    