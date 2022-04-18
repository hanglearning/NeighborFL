import math
from copy import deepcopy
import random
import os
from haversine import haversine, Unit
from keras.models import load_model

class Detector:
    preserve_historical_models = 0
    def __init__(self, id, radius=None,k=None,latitude=None, longitude=None, direction=None, num_neighbors_try=1, add_heuristic=1, epsilon=0.2) -> None:
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
        self.to_compare_fav_neighbors_fl_predictions = None
        self.neighbors = [] # candidate neighbors
        self.fav_neighbors = []
        self.tried_neighbors = []
        self.neighbor_to_last_accumulate = {}
        self.neighbor_to_accumulate_interval = {}
        self.neighbors_to_rep_score = {}
        self.num_neighbors_try = num_neighbors_try
        self.add_heuristic = add_heuristic
        self.epsilon = epsilon # if model itself is worse than last comm_round, roll a dice and kick a neighbor with the lowest reputation score, which is accumulated by error diffence. The larger the neighbor's model brings down the error, the better the neighbor
            
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
        elif self.add_heuristic == 2:
            # add neighbors randomly
            self.neighbors = random.shuffle(self.neighbors)
            
    def init_models(self, global_model_0_path):
        self.stand_alone_model_path = global_model_0_path
        self.naive_fl_global_model_path = global_model_0_path
        self.fav_neighbors_fl_agg_model_path = global_model_0_path
    
    def get_stand_alone_model(self):
        return load_model(self.stand_alone_model_path)
    
    def get_naive_fl_local_model(self):
        return load_model(self.naive_fl_local_model_path)
    
    def get_last_naive_fl_global_model(self):
        return load_model(self.naive_fl_global_model_path)
    
    def get_fav_neighbors_fl_local_model(self):
        return load_model(self.fav_neighbors_fl_local_model_path)
    
    def get_last_fav_neighbors_fl_agg_model(self):
        return load_model(self.fav_neighbors_fl_agg_model_path)
        
    def update_and_save_stand_alone_model(self, new_model, comm_round, stand_alone_model_path):
        os.makedirs(f'{stand_alone_model_path}/{self.id}', exist_ok=True)
        new_model_path = f'{stand_alone_model_path}/{self.id}/comm_{comm_round}.h5'
        new_model.save(new_model_path)
        self.stand_alone_model_path = new_model_path
        self.delete_historical_models(f'{stand_alone_model_path}/{self.id}', comm_round)
    
    def update_and_save_naive_fl_local_model(self, new_model, comm_round, naive_fl_local_model_path):
        os.makedirs(f'{naive_fl_local_model_path}/{self.id}', exist_ok=True)
        new_model_path = f'{naive_fl_local_model_path}/{self.id}/comm_{comm_round}.h5'
        new_model.save(new_model_path)
        self.naive_fl_local_model_path = new_model_path
        self.delete_historical_models(f'{naive_fl_local_model_path}/{self.id}', comm_round)
    
    def update_fl_global_model(self, comm_round, naive_fl_global_model_path):
        self.naive_fl_global_model_path = f'{naive_fl_global_model_path}/comm_{comm_round}.h5'
        
    def update_and_save_fav_neighbors_fl_local_model(self, new_model, comm_round, fav_neighbors_fl_local_model_path):
        os.makedirs(f'{fav_neighbors_fl_local_model_path}/{self.id}', exist_ok=True)
        new_model_path = f'{fav_neighbors_fl_local_model_path}/{self.id}/comm_{comm_round}.h5'
        new_model.save(new_model_path)
        self.fav_neighbors_fl_local_model_path = new_model_path
        self.delete_historical_models(f'{fav_neighbors_fl_local_model_path}/{self.id}', comm_round)
        
    @classmethod
    def save_fl_global_model(cls, new_model, comm_round, naive_fl_global_model_path):
        new_model.save(f'{naive_fl_global_model_path}/comm_{comm_round}.h5')
        cls.delete_historical_models(f'{naive_fl_global_model_path}', comm_round)
        
    def update_and_save_fav_neighbors_fl_agg_model(self, new_model, comm_round, fav_neighbors_fl_agg_model_path):
        os.makedirs(f'{fav_neighbors_fl_agg_model_path}/{self.id}', exist_ok=True)
        new_model_path = f'{fav_neighbors_fl_agg_model_path}/{self.id}/comm_{comm_round}.h5'
        new_model.save(new_model_path)
        self.fav_neighbors_fl_agg_model_path = new_model_path
        self.delete_historical_models(f'{fav_neighbors_fl_agg_model_path}/{self.id}', comm_round)
    
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
       
    
            