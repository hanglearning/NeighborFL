import math
from copy import deepcopy
import random
import os

class Detector:
    def __init__(self, id, dataset, radius=None,k=None,latitude=None, longitude=None, direction=None, num_neighbors_try=1, add_heuristic=1, epsilon=0.2, preserve_historical_model_files=0) -> None:
        self.id = id
        self.loc = (latitude, longitude)
        self.direction = direction
        self._dataset = dataset # read in csv
        self._current_round_X_test = None
        self._current_round_y_true = None
        self.radius = radius # treat detectors within this radius as neighbors, unit miles. Default to None - consider every participating detector
        # 3 models to report and compare
        self.k = k # set maximum number of possible fav_neighbors. Can be used with radius
        self.stand_alone_model = None
        self.naive_fl_local_model = None # CCGrid
        self.naive_fl_model = None # CCGrid
        self.fav_neighbors_fl_local_model = None
        self.fav_neighbors_fl_model = None
        self.fav_neighbors_fl_predictions = None
        self.to_compare_fav_neighbors_fl_predictions = None
        self.neighbors = [] # candidate neighbors
        self.fav_neighbors = []
        self.tried_neighbors = []
        self.neighbor_to_last_accumulate = {}
        self.neighbor_to_accumulate_interval = {}
        self.neighbors_to_rep_score = {}
        self.num_neighbors_try = num_neighbors_try
        self.add_heuristic = add_heuristic
        self.epsilon = epsilon # if model itself is worse than last round, roll a dice and kick a neighbor with the lowest reputation score, which is accumulated by error diffence. The larger the neighbor's model brings down the error, the better the neighbor
        self.preserve_historical_models = preserve_historical_model_files
            
    def assign_neighbors(self, list_of_detectors):
        for detector_id, detector in list_of_detectors.items():
            if detector == self:
                continue
            distance = math.sqrt((self.loc[0] - detector.loc[0]) ** 2 + (self.loc[1] - detector.loc[1]) ** 2)
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
            
    def init_models(self, global_model_0):
        self.stand_alone_model = deepcopy(global_model_0)
        self.naive_fl_model = deepcopy(global_model_0)
        self.fav_neighbors_fl_model = deepcopy(global_model_0)
        
    def update_and_save_stand_alone_model(self, new_model, round, stand_alone_model_path):
        self.stand_alone_model = new_model
        self.stand_alone_model.save(f'{stand_alone_model_path}/{self.id}/comm_{round}.h5')
        self.delete_historical_models(f'{stand_alone_model_path}/{self.id}', round)
    
    def update_naive_fl_local_model(self, new_model):
        self.naive_fl_local_model = new_model
        
    def update_fav_neighbors_fl_local_model(self, new_model):
        self.fav_neighbors_fl_local_model = new_model
    
    def update_and_save_naive_fl_model(self, new_model, round, naive_fl_model_path):
        self.naive_fl_model = new_model
        self.naive_fl_model.save(f'{naive_fl_model_path}/{self.id}/comm_{round}.h5')
        self.delete_historical_models(f'{naive_fl_model_path}/{self.id}', round)
        
    def update_and_save_fav_neighbors_fl_model(self, new_model, round, fav_neighbors_fl_model_path):
        self.fav_neighbors_fl_model = new_model
        self.fav_neighbors_fl_model.save(f'{fav_neighbors_fl_model_path}/{self.id}/comm_{round}.h5')
        self.delete_historical_models(f'{fav_neighbors_fl_model_path}/{self.id}', round)
        
    def delete_historical_models(self, model_root_path, round):
        if not self.preserve_historical_models:
            filelist = [f for f in os.listdir(model_root_path) if not f.endswith(f'comm_{round}.h5') and not f.endswith(f'comm_{round-1}.h5')]
            for f in filelist:
                os.remove(os.path.join(model_root_path, f))
    
    def get_dataset(self):
        return self._dataset
    
    def set_X_test(self, X_test):
        self._current_round_X_test = X_test
        
    def set_y_true(self, y_true):
        self._current_round_y_true = y_true
        
    def get_X_test(self):
        return self._current_round_X_test
        
    def get_y_true(self):
        return self._current_round_y_true
       
    
            