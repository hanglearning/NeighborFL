import math
from copy import deepcopy
import random
import os
from haversine import haversine, Unit
from tensorflow.keras.models import load_model

class Detector:
    preserve_historical_models = 0
    logs_dirpath = None
    # if it's 1st comm round or whenever starting a new Monday, need to flush memory and start with 24 training samples
    reset_mem = True
    training_data_starting_index = 0
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
        self.same_dir_fl_local_model_path = None
        self.same_dir_fl_global_model_path = None

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

    def assign_neighbors(self, list_of_detectors):
        for detector_id, detector in list_of_detectors.items():
            if detector == self:
                continue
            distance = haversine(self.loc, detector.loc, unit='mi')
            if not self.radius:
                # treat all participants as neighbors
                self.neighbors.append((detector, distance))
            else:
                if distance <= self.radius:
                    self.neighbors.append((detector, distance))
        if self.add_heuristic == 1:
            # add neighbors with close to far heuristic
            self.neighbors = sorted(self.neighbors, key = lambda x: x[1])
        elif self.add_heuristic == 2:
            # add neighbors randomly
            self.neighbors = random.shuffle(self.neighbors)
        print(f"Detector {self.id} has detected {len(self.neighbors)} potential neighbors within radius {self.radius} miles.")
            
    def init_models(self, global_model_0_path):
        self.stand_alone_model_path = global_model_0_path
        self.naive_fl_global_model_path = global_model_0_path
        self.fav_neighbors_fl_agg_model_path = global_model_0_path
        self.same_dir_fl_global_model_path = global_model_0_path

    def get_stand_alone_model(self):
        return load_model(f'{self.logs_dirpath}/{self.stand_alone_model_path}', compile = False)
    
    def get_naive_fl_local_model(self):
        return load_model(f'{self.logs_dirpath}/{self.naive_fl_local_model_path}', compile = False)
    
    def get_last_naive_fl_global_model(self):
        return load_model(f'{self.logs_dirpath}/{self.naive_fl_global_model_path}', compile = False)

    def get_same_dir_fl_local_model(self):
        return load_model(f'{self.logs_dirpath}/{self.same_dir_fl_local_model_path}', compile = False)
    
    def get_last_same_dir_fl_global_model(self):
        return load_model(f'{self.logs_dirpath}/{self.same_dir_fl_global_model_path}', compile = False)
    
    def get_fav_neighbors_fl_local_model(self):
        return load_model(f'{self.logs_dirpath}/{self.fav_neighbors_fl_local_model_path}', compile = False)
    
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

    @classmethod
    def save_N_global_model(cls, new_model, comm_round, N_dir_fl_global_model_path):
        new_model.save(f'{cls.logs_dirpath}/{N_dir_fl_global_model_path}/comm_{comm_round}.h5')
        cls.delete_historical_models(f'{cls.logs_dirpath}/{N_dir_fl_global_model_path}', comm_round)

    @classmethod
    def save_S_global_model(cls, new_model, comm_round, S_dir_fl_global_model_path):
        new_model.save(f'{cls.logs_dirpath}/{S_dir_fl_global_model_path}/comm_{comm_round}.h5')
        cls.delete_historical_models(f'{cls.logs_dirpath}/{S_dir_fl_global_model_path}', comm_round)
    
    def update_fl_global_model(self, comm_round, naive_fl_global_model_path):
        self.naive_fl_global_model_path = f'{naive_fl_global_model_path}/comm_{comm_round}.h5'

    def update_same_dir_fl_global_model(self, comm_round, same_dir_fl_global_model_path):
        self.same_dir_fl_global_model_path = f'{same_dir_fl_global_model_path}/comm_{comm_round}.h5'
        
    def update_and_save_model(self, new_model, comm_round, model_folder_name):
        os.makedirs(f'{self.logs_dirpath}/{model_folder_name}/{self.id}', exist_ok=True)
        new_model_path = f'{model_folder_name}/{self.id}/comm_{comm_round}.h5'
        new_model.save(f'{self.logs_dirpath}/{new_model_path}')
        if model_folder_name == 'stand_alone': 
            self.stand_alone_model_path = new_model_path
        elif model_folder_name == 'naive_fl_local': 
            self.naive_fl_local_model_path = new_model_path
        elif model_folder_name == 'same_dir_fl_local': 
            self.same_dir_fl_local_model_path = new_model_path
        elif model_folder_name == 'fav_neighbors_fl_local': 
            self.fav_neighbors_fl_local_model_path = new_model_path
        elif model_folder_name == 'fav_neighbors_fl_agg': 
            self.fav_neighbors_fl_agg_model_path = new_model_path
        elif model_folder_name == 'tried_fav_neighbors_fl_agg': 
            self.tried_fav_neighbors_fl_agg_model_path = new_model_path
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
       
    
            