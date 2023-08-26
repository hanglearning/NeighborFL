import math
from copy import deepcopy
import random
import os
from haversine import haversine, Unit
from tensorflow.keras.models import load_model

class Device:
    preserve_historical_models = 0
    logs_dirpath = None
    def __init__(self, id, radius=None,latitude=None, longitude=None, direction=None, num_neighbors_try=1, add_heuristic=1) -> None:
        self.id = id
        self.loc = (latitude, longitude)
        self.direction = direction
        self.radius = radius # treat devices within this radius as candidate neighbors, unit miles. Default to None - consider every participating device

        # baseline
        self.central_model_path = None

        # pure FedAvg
        self.naive_fl_local_model_path = None # CCGrid
        self.naive_fl_global_model_path = None # CCGrid

        # same dir pure FedAvg - NOT USED IN PAPER
        self.same_dir_fl_local_model_path = None
        self.same_dir_fl_global_model_path = None

        # radius pure FedAvg
        self.radius_naive_fl_local_model_path = None 
        self.radius_naive_fl_agg_model_path = None 

        # radius same dir pure FedAvg - NOT USED IN PAPER
        self.radius_same_dir_fl_local_model_path = None
        self.radius_same_dir_fl_agg_model_path = None

        # fav_neighbor
        self.neighbor_fl_local_model_path = None
        self.neighbor_fl_agg_model_path = None # aggregated


        self.neighbor_fl_predictions = None
        self.eval_neighbor_fl_agg_model_path = None # aggregated

        self.candidate_neighbors = []
        self.fav_neighbors = []
        self.eval_neighbors = []
        self.neighbor_to_last_accumulate = {}
        self.neighbor_to_accumulate_interval = {}
        self.neighbors_to_rep_score = {}
        self.num_neighbors_try = num_neighbors_try
        self.add_heuristic = add_heuristic
        self.neighbor_fl_error_records = []
        self.has_added_neigbor = False # used as one of the conditions to determine whether to kick a fav neighbor

    def assign_candidate_neighbors(self, list_of_devices):
        for device_id, device in list_of_devices.items():
            if device == self:
                continue
            distance = haversine(self.loc, device.loc, unit='mi')
            if not self.radius:
                # treat all participants as neighbors
                self.candidate_neighbors.append((device, distance))
            else:
                if distance <= self.radius:
                    self.candidate_neighbors.append((device, distance))
        if self.add_heuristic == 1:
            # add neighbors with close to far heuristic
            self.candidate_neighbors = sorted(self.candidate_neighbors, key = lambda x: x[1])
        elif self.add_heuristic == 2:
            # add neighbors randomly
            self.candidate_neighbors = random.shuffle(self.candidate_neighbors)
        print(f"Device {self.id} has detected {len(self.candidate_neighbors)} potential neighbors within radius {self.radius} miles.")
            
    def init_models(self, global_model_0_or_pretrained_model_path):

        # baseline
        self.central_model_path = global_model_0_or_pretrained_model_path

        # pure FedAvg
        self.naive_fl_global_model_path = global_model_0_or_pretrained_model_path

        # same dir pure FedAvg - NOT USED IN PAPER
        self.same_dir_fl_global_model_path = global_model_0_or_pretrained_model_path

        # radius pure FedAvg
        self.radius_naive_fl_agg_model_path = global_model_0_or_pretrained_model_path

        # radius same dir pure FedAvg - NOT USED IN PAPER
        self.radius_same_dir_fl_agg_model_path = global_model_0_or_pretrained_model_path

        # fav_neighbors
        self.neighbor_fl_agg_model_path = global_model_0_or_pretrained_model_path
        

    # standalone
    def get_central_model(self):
        return load_model(f'{self.logs_dirpath}/{self.central_model_path}', compile = False)
    
    # pure FedAvg
    def get_naive_fl_local_model(self):
        return load_model(f'{self.logs_dirpath}/{self.naive_fl_local_model_path}', compile = False)
    
    def get_last_naive_fl_global_model(self):
        return load_model(f'{self.logs_dirpath}/{self.naive_fl_global_model_path}', compile = False)

    # same dir pure FedAvg - NOT USED IN PAPER
    def get_same_dir_fl_local_model(self):
        return load_model(f'{self.logs_dirpath}/{self.same_dir_fl_local_model_path}', compile = False)
    
    def get_last_same_dir_fl_global_model(self):
        return load_model(f'{self.logs_dirpath}/{self.same_dir_fl_global_model_path}', compile = False)
    
    # radius pure FedAvg
    def get_radius_naive_fl_local_model(self):
        return load_model(f'{self.logs_dirpath}/{self.radius_naive_fl_local_model_path}', compile = False)
    
    def get_last_radius_naive_fl_agg_model(self):
        return load_model(f'{self.logs_dirpath}/{self.radius_naive_fl_agg_model_path}', compile = False)

    # same dir pure FedAvg - NOT USED IN PAPER
    def get_radius_same_dir_fl_local_model(self):
        return load_model(f'{self.logs_dirpath}/{self.radius_same_dir_fl_local_model_path}', compile = False)
    
    def get_last_radius_same_dir_fl_agg_model(self):
        return load_model(f'{self.logs_dirpath}/{self.radius_same_dir_fl_agg_model_path}', compile = False)
    
    # fav_neighbors
    def get_neighbor_fl_local_model(self):
        return load_model(f'{self.logs_dirpath}/{self.neighbor_fl_local_model_path}', compile = False)
    
    def get_last_neighbor_fl_agg_model(self):
        return load_model(f'{self.logs_dirpath}/{self.neighbor_fl_agg_model_path}', compile = False)
    
    def get_eval_neighbor_fl_agg_model(self):
        model_path = f'{self.logs_dirpath}/{self.eval_neighbor_fl_agg_model_path}'
        if not os.path.isfile(model_path):
            return None
        return load_model(model_path, compile = False)
    
    # pure FedAvg
    @classmethod
    def save_fl_global_model(cls, new_model, comm_round, naive_fl_global_model_path):
        new_model.save(f'{cls.logs_dirpath}/{naive_fl_global_model_path}/comm_{comm_round}.h5')
        cls.delete_historical_models(f'{cls.logs_dirpath}/{naive_fl_global_model_path}', comm_round)

    # same dir pure FedAvg - NOT USED IN PAPER
    @classmethod
    def save_N_global_model(cls, new_model, comm_round, N_dir_fl_global_model_path):
        new_model.save(f'{cls.logs_dirpath}/{N_dir_fl_global_model_path}/comm_{comm_round}.h5')
        cls.delete_historical_models(f'{cls.logs_dirpath}/{N_dir_fl_global_model_path}', comm_round)

    @classmethod
    def save_S_global_model(cls, new_model, comm_round, S_dir_fl_global_model_path):
        new_model.save(f'{cls.logs_dirpath}/{S_dir_fl_global_model_path}/comm_{comm_round}.h5')
        cls.delete_historical_models(f'{cls.logs_dirpath}/{S_dir_fl_global_model_path}', comm_round)
    
    
    # pure FedAvg
    def update_naive_fl_global_model(self, comm_round, naive_fl_global_model_path):
        self.naive_fl_global_model_path = f'{naive_fl_global_model_path}/comm_{comm_round}.h5'

    # same dir pure FedAvg - NOT USED IN PAPER
    def update_same_dir_fl_global_model(self, comm_round, same_dir_fl_global_model_path):
        self.same_dir_fl_global_model_path = f'{same_dir_fl_global_model_path}/comm_{comm_round}.h5'

    
        
    def update_and_save_model(self, new_model, comm_round, model_folder_name):
        os.makedirs(f'{self.logs_dirpath}/{model_folder_name}/{self.id}', exist_ok=True)
        new_model_path = f'{model_folder_name}/{self.id}/comm_{comm_round}.h5'
        new_model.save(f'{self.logs_dirpath}/{new_model_path}')
        # baseline
        if model_folder_name == 'central': 
            self.central_model_path = new_model_path

        # pure FedAvg
        elif model_folder_name == 'naive_fl_local': 
            self.naive_fl_local_model_path = new_model_path

        # same dir pure FedAvg - NOT USED IN PAPER
        elif model_folder_name == 'same_dir_fl_local': 
            self.same_dir_fl_local_model_path = new_model_path
        
        # pure radius FedAvg
        elif model_folder_name == 'radius_naive_fl_local': 
            self.radius_naive_fl_local_model_path = new_model_path

        elif model_folder_name == 'radius_naive_fl_agg': 
            self.radius_naive_fl_agg_model_path = new_model_path

        # same dir pure FedAvg - NOT USED IN PAPER
        elif model_folder_name == 'radius_same_dir_fl_local': 
            self.radius_same_dir_fl_local_model_path = new_model_path

        elif model_folder_name == 'radius_same_dir_fl_agg': 
            self.radius_same_dir_fl_agg_model_path = new_model_path

        # fav_neighbors
        elif model_folder_name == 'neighbor_fl_local': 
            self.neighbor_fl_local_model_path = new_model_path
        elif model_folder_name == 'neighbor_fl_agg': 
            self.neighbor_fl_agg_model_path = new_model_path
        elif model_folder_name == 'eval_neighbor_fl_agg': 
            self.eval_neighbor_fl_agg_model_path = new_model_path
        self.delete_historical_models(f'{self.logs_dirpath}/{model_folder_name}/{self.id}', comm_round)
    
    @classmethod
    def delete_historical_models(cls, model_root_path, comm_round):
        if not cls.preserve_historical_models:
            filelist = [f for f in os.listdir(model_root_path) if not f.endswith(f'comm_{comm_round}.h5') and not f.endswith(f'comm_{comm_round-1}.h5')]
            for f in filelist:
                os.remove(os.path.join(model_root_path, f))
    
       
    
            