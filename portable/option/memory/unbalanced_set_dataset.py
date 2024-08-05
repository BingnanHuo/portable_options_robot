import torch 
import numpy as np 
import math 
import os 
import pickle 

class UnbalancedSetDataset():
    def __init__(self,
                 batchsize=64,
                 unlabelled_batchsize=None,
                 max_size=100000,
                 ):
        self.batchsize = batchsize
        if unlabelled_batchsize is not None:
            self.dynamic_unlabelled_batchsize = False
            self.unlabelled_batchsize = unlabelled_batchsize
        else:
            self.dynamic_unlabelled_batchsize = True
            self.unlabelled_batchsize = 0
        self.max_size = max_size
        
        self.data = torch.from_numpy(np.array([]))
        self.unlabelled_data = torch.from_numpy(np.array([]))
        self.labels = torch.from_numpy(np.array([]))
        
        self.data_length = 0
        self.unlabelled_data_length = 0
        self.counter = 0
        self.num_batches = 0
        
        self.shuffled_indices = None
    
    @staticmethod
    def transform(x):
        if torch.max(x) > 1:
            return (x/255.0).float()
        else:
            return x
    
    def get_equal_class_weight(self):
        num_positive = torch.sum(self.labels)
        num_negative = self.data_length - num_positive
        
        return [self.data_length/num_negative, 
                self.data_length/num_positive]
    
    def reset(self):
        self.data = torch.from_numpy(np.array([]))
        self.labels = torch.from_numpy(np.array([]))
        self.counter = 0
        self.num_batches = 0
        self.shuffled_indices = None
    
    def reset_memory(self):
        self.data = torch.from_numpy(np.array([]))
        self.labels = torch.from_numpy(np.array([]))
    
    def set_transform_function(self, transform):
        self.transform = transform
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        
        with open(os.path.join(path, "data.pkl")) as f:
            pickle.dump(self.data, f)
        
        with open(os.path.join(path, "label.pkl")) as f:
            pickle.dump(self.labels, f)
    
    def load(self, path):
        data_file = os.path.join(path, "data.pkl")
        label_file = os.path.join(path, "label.pkl")
        if not os.path.exists(data_file):
            print("[UnbalancedSetDataset] No data found.")
            return
        if not os.path.exists(label_file):
            print("[UnbalancedSetDataset] No labels found.")
            return 
        with open(data_file, "rb") as f:
            self.data = pickle.load(f)
        self.data_length = len(self.data)
        with open(label_file, "rb") as f:
            self.labels = pickle.load(f)
        
        self._set_batch_num()
        self.shuffle()
    
    def shuffle(self):
        self.shuffled_indices = np.random.permutation(self.data_length)
    
    def _set_batch_num(self):
        self.num_batches = math.ceil(
            self.data_length/self.batchsize
        )
    
    def add_true_files(self, file_list):
        for file in file_list:
            data = np.load(file, allow_pickle=True)
            data = torch.from_numpy(data)
            if torch.max(data) <= 1:
                data = data*255
            data = data.int()
            data = data.squeeze()
            labels = torch.ones(len(data), dtype=torch.int8)
            self.data = self.concatenate(self.data, data)
            self.labels = self.concatenate(self.labels, labels)
            assert len(self.data) == len(self.labels)
            self.data_length = len(self.data)
            self._set_batch_num()
            self.counter = 0
            
            self.shuffle()
    
    def add_false_files(self, file_list):
        for file in file_list:
            data = np.load(file, allow_pickle=True)
            data = torch.from_numpy(data)
            if torch.max(data) <= 1:
                data = data*255
            data = data.int()
            data = data.squeeze()
            labels = torch.zeros(len(data), dtype=torch.int8)
            self.data = self.concatenate(self.data, data)
            self.labels = self.concatenate(self.labels, labels)
            assert len(self.data) == len(self.labels)
            self.data_length = len(self.data)
            self._set_batch_num()
            self.counter = 0
            
            self.shuffle()
    
    def get_batch(self, shuffle_batch=True):
        data = []
        labels = []
        if (self.index() + self.batchsize) > self.data_length:
            data = self.data[self.index():]
            labels = self.labels[self.index():]
        else:
            data = self.data[self.index():self.index() + self.batchsize]
            labels = self.labels[self.index():self.index() + self.batchsize]
        
        data = self.transform(data)
        
        self.counter += 1
        
        return data, labels
    
    def index(self):
        return (self.counter*self.batchsize)%self.data_length
    
    @staticmethod
    def concatenate(arr1, arr2):

        if len(arr1) == 0:
            return arr2
        else:
            return torch.cat((arr1, arr2), axis=0)