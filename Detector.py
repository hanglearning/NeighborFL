import math
from copy import deepcopy
import random
import os
from haversine import haversine, Unit
from tensorflow.keras.models import load_model

from itertools import chain, combinations
import numpy as np
import pickle

from time import time

def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func

class Detector:
    preserve_historical_models = 0
    logs_dirpath = None
    # brute_force
    dp_brute_force_models = {}
    brute_force_combo_error_records = {}
    brute_force_best_neighbors = {}
    def __init__(self, id, radius=None,k=None,latitude=None, longitude=None, direction=None, num_neighbors_try=1, add_heuristic=1) -> None:
        self.id = id
        self.loc = (latitude, longitude)
        self.direction = direction
        # self._dataset = dataset # save time while saving detector object to file
        self._current_comm_round_X_test = None
        self._current_comm_round_y_true = None
        self.radius = radius # treat detectors within this radius as neighbors, unit miles. Default to None - consider every participating detector
        # 3 models to report and compare
        self.k = k # set maximum number of possible fav_neighbors. Can be used with radius
        self.stand_alone_model_path = None
        self.naive_fl_local_model_path = None # CCGrid
        self.naive_fl_global_model_path = None # CCGrid
        self.fav_neighbors_fl_local_model_path = None
        self.fav_neighbors_fl_agg_model_path = None # aggregated
        self.fav_neighbors_fl_predictions = None
        self.tried_fav_neighbors_fl_agg_model_path = None # aggregated
        self.tried_fav_neighbors_fl_predictions = None
        self.neighbors = [] # candidate neighbors
        self.fav_neighbors = []
        self.tried_neighbors = []
        self.neighbor_to_last_accumulate = {}
        self.neighbor_to_accumulate_interval = {}
        self.neighbors_to_rep_score = {}
        self.num_neighbors_try = num_neighbors_try
        self.add_heuristic = add_heuristic
        self.neighbor_fl_error_records = []
        # brute-force
        self.brute_force_fl_local_model_path = None
        self.brute_force_fl_agg_model_path = None
        self.brute_force_neighbor_combinations = None
        self.brute_force_all_models_predictions = {}
        self.neighbor_to_distance = {}
        
    def assign_neighbors(self, list_of_detectors):
        for detector_id, detector in list_of_detectors.items():
            if detector == self:
                continue
            distance = haversine(self.loc, detector.loc)
            if not self.radius:
                # treat all participants as neighbors
                self.neighbors.append((detector, distance))
            else:
                if distance <= self.radius:
                    self.neighbors.append((detector, distance))
        if self.add_heuristic == 1:
            # add neighbors with close to far heuristic
            self.neighbors = sorted(self.neighbors, key = lambda x: x[1])
            neighbor_order = 1
            for neighbor in self.neighbors:
                # brute-force helper
                self.neighbor_to_distance[neighbor[0].id] = neighbor_order
                neighbor_order += 1
        elif self.add_heuristic == 2:
            # add neighbors randomly
            self.neighbors = random.shuffle(self.neighbors)
            
    def set_brute_force_neighbor_combinations(self, list_of_detectors):
        detector_ids = [d for d in list(list_of_detectors.keys()) if d != self.id]
        # https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
        s = list(detector_ids)
        self.brute_force_neighbor_combinations = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
    
    @timer_func
    def get_all_possible_models_predictions(self, create_model, model_units, model_configs, list_of_detectors, scaler):
        for combo in self.brute_force_neighbor_combinations:
            detectors_in_this_model = set([self.id])
            for neighbor in combo:
                detectors_in_this_model.add(neighbor)
            detectors_in_this_model = frozenset(detectors_in_this_model)
            if detectors_in_this_model in self.dp_brute_force_models:
                temp_model = self.dp_brute_force_models[detectors_in_this_model]
            else:
                temp_model = create_model(model_units, model_configs)
                model_weights = [self.get_brute_force_fl_local_model().get_weights()] 
                for neighbor in combo:
                    model_weights.append(list_of_detectors[neighbor].get_brute_force_fl_local_model().get_weights())
                temp_model.set_weights(np.mean(model_weights, axis=0))
                # use dp_brute_force_models to record model for other detectors to speed up execution
                self.dp_brute_force_models[frozenset(detectors_in_this_model)] = temp_model
            temp_predictions = temp_model.predict(self.get_X_test())
            temp_predictions = scaler.inverse_transform(temp_predictions.reshape(-1, 1)).reshape(1, -1)[0]
            self.brute_force_all_models_predictions[combo] = temp_predictions
        # return self.brute_force_all_models_predictions
    
    @timer_func
    def get_best_brute_force_model(self, create_model, model_units, model_configs, get_error, y_true, list_of_detectors, comm_round):
        all_combos_error_records = {}
        for combo, pred in self.brute_force_all_models_predictions.items():
            all_combos_error_records[combo] = get_error(y_true, pred)
        # sort by errors
        combo_pred_tuples = [(combo, pred) for combo, pred in sorted(all_combos_error_records.items(), key=lambda x: x[1])]
        # add distance order
        combo_pred_tuples_with_distance_order = [([(neighbor, self.neighbor_to_distance[neighbor]) for neighbor in combo_pred_tuple[0]], combo_pred_tuple[1]) for combo_pred_tuple in combo_pred_tuples]
        
        # recreate best fav model
        best_combo = combo_pred_tuples_with_distance_order[0][0]
        best_pred = self.brute_force_all_models_predictions[combo_pred_tuples[0][0]]
        
        if self.id in self.brute_force_combo_error_records:
            self.brute_force_combo_error_records[self.id].append(combo_pred_tuples_with_distance_order)
        else:
            self.brute_force_combo_error_records[self.id] = [combo_pred_tuples_with_distance_order]
            
        if self.id in self.brute_force_best_neighbors:
            self.brute_force_best_neighbors[self.id].append(best_combo)
        else:
            self.brute_force_best_neighbors[self.id] = [best_combo]
        
        model_weights = [self.get_last_brute_force_fl_local_model(comm_round).get_weights()]
        temp_model = create_model(model_units, model_configs)
        for best_neighbor_iter in best_combo:
            model_weights.append(list_of_detectors[best_neighbor_iter[0]].get_last_brute_force_fl_local_model(comm_round).get_weights())
        temp_model.set_weights(np.mean(model_weights, axis=0))
        return temp_model, best_pred
        
    def init_models(self, global_model_0_path):
        self.stand_alone_model_path = global_model_0_path
        self.naive_fl_global_model_path = global_model_0_path
        self.fav_neighbors_fl_agg_model_path = global_model_0_path
        self.brute_force_fl_agg_model_path = global_model_0_path
    
    def get_stand_alone_model(self):
        return load_model(f'{self.logs_dirpath}/{self.stand_alone_model_path}', compile = False)
    
    def get_naive_fl_local_model(self):
        return load_model(f'{self.logs_dirpath}/{self.naive_fl_local_model_path}', compile = False)
    
    def get_last_naive_fl_global_model(self):
        return load_model(f'{self.logs_dirpath}/{self.naive_fl_global_model_path}', compile = False)
    
    def get_fav_neighbors_fl_local_model(self):
        return load_model(f'{self.logs_dirpath}/{self.fav_neighbors_fl_local_model_path}', compile = False)
    
    # brute-force helper func
    def get_brute_force_fl_local_model(self):
        return load_model(f'{self.logs_dirpath}/{self.brute_force_fl_local_model_path}', compile = False)
    
    def get_brute_force_fl_agg_model(self):
        return load_model(f'{self.logs_dirpath}/{self.brute_force_fl_agg_model_path}', compile = False)
    
    def get_last_brute_force_fl_local_model(self, comm_round):
        return load_model(f'{self.logs_dirpath}/brute_force_fl_local/{self.id}/comm_{comm_round-1}.h5', compile = False)
    
    def get_last_fav_neighbors_fl_agg_model(self):
        return load_model(f'{self.logs_dirpath}/{self.fav_neighbors_fl_agg_model_path}', compile = False)
    
    def get_tried_fav_neighbors_fl_agg_model(self):
        model_path = f'{self.logs_dirpath}/{self.tried_fav_neighbors_fl_agg_model_path}'
        if not os.path.isfile(model_path):
            return None
        return load_model(model_path, compile = False)
        
    @classmethod
    def save_fl_global_model(cls, new_model, comm_round, naive_fl_global_model_path):
        new_model.save(f'{cls.logs_dirpath}/{naive_fl_global_model_path}/comm_{comm_round}.h5')
        cls.delete_historical_models(f'{cls.logs_dirpath}/{naive_fl_global_model_path}', comm_round)
    
    def update_fl_global_model(self, comm_round, naive_fl_global_model_path):
        self.naive_fl_global_model_path = f'{naive_fl_global_model_path}/comm_{comm_round}.h5'
        
    def update_and_save_model(self, new_model, comm_round, model_folder_name):
        os.makedirs(f'{self.logs_dirpath}/{model_folder_name}/{self.id}', exist_ok=True)
        new_model_path = f'{model_folder_name}/{self.id}/comm_{comm_round}.h5'
        new_model.save(f'{self.logs_dirpath}/{new_model_path}')
        if model_folder_name == 'stand_alone': 
            self.stand_alone_model_path = new_model_path
        elif model_folder_name == 'naive_fl_local': 
            self.naive_fl_local_model_path = new_model_path
        elif model_folder_name == 'fav_neighbors_fl_local': 
            self.fav_neighbors_fl_local_model_path = new_model_path
        elif model_folder_name == 'fav_neighbors_fl_agg': 
            self.fav_neighbors_fl_agg_model_path = new_model_path
        elif model_folder_name == 'tried_fav_neighbors_fl_agg': 
            self.tried_fav_neighbors_fl_agg_model_path = new_model_path
        elif model_folder_name == 'brute_force_fl_local': 
            self.brute_force_fl_local_model_path = new_model_path
        elif model_folder_name == 'brute_force_fl_agg': 
            self.brute_force_fl_agg_model_path = new_model_path
        self.delete_historical_models(f'{self.logs_dirpath}/{model_folder_name}/{self.id}', comm_round)
    
    @classmethod
    def delete_historical_models(cls, model_root_path, comm_round):
        if not cls.preserve_historical_models:
            filelist = [f for f in os.listdir(model_root_path) if not f.endswith(f'comm_{comm_round}.h5') and not f.endswith(f'comm_{comm_round-1}.h5')]
            for f in filelist:
                os.remove(os.path.join(model_root_path, f))
    
    # def get_dataset(self):
    #     return self._dataset
    
    def set_X_test(self, X_test):
        self._current_comm_round_X_test = X_test
        
    def set_y_true(self, y_true):
        self._current_comm_round_y_true = y_true
        
    def get_X_test(self):
        return self._current_comm_round_X_test
        
    def get_y_true(self):
        return self._current_comm_round_y_true
       
    
            