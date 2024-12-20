import datetime
import logging
import os
import pickle

import gin
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from portable.option.divdis.divdis_classifier import DivDisClassifier
#from portable.option.divdis.divdis_classifier_no_div import DivDisClassifier
from portable.option.memory import SetDataset
from portable.option.memory.unbalanced_set_dataset import UnbalancedSetDataset
from portable.utils.utils import set_seed



def transform(x):
    return x

def load_image(file_path):
    preproc_pipeline = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ])
    try:
        image = Image.open(file_path).convert('RGB')
        image = preproc_pipeline(image)
        image_tensor = torch.tensor(np.array(image), dtype=torch.uint8).permute(2, 0, 1)
        return image_tensor.unsqueeze(0)
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None

@gin.configurable 
class DivDisClassifierExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 seed,
                 use_gpu,
                 
                 classifier_head_num,
                 classifier_learning_rate,
                 classifier_num_classes,
                 classifier_diversity_weight,
                 classifier_l2_reg_weight,
                 classifier_train_epochs,
                 classifier_model_name):
        
        self.seed = seed 
        self.base_dir = base_dir
        self.experiment_name = experiment_name 
        
        set_seed(seed)
        
        self.base_dir = os.path.join(base_dir, experiment_name, str(seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        
        self.classifier = DivDisClassifier(use_gpu=use_gpu,
                                           log_dir=self.log_dir,
                                           head_num=classifier_head_num,
                                           learning_rate=classifier_learning_rate,
                                           num_classes=classifier_num_classes,
                                           diversity_weight=classifier_diversity_weight,
                                           l2_reg_weight=classifier_l2_reg_weight,
                                           model_name=classifier_model_name)
        self.classifier.dataset.set_transform_function(transform)
        self.classifier.dataset.set_load_data_function(load_image)
        
        self.train_epochs = classifier_train_epochs
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        log_file = os.path.join(self.log_dir,
                                "{}.log".format(datetime.datetime.now()))
        
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        
        logging.info("[experiment] Beginning experiment {} seed {}".format(self.experiment_name, self.seed))
        logging.info("======== HYPERPARAMETERS ========")
        logging.info("Seed: {}".format(seed))
        logging.info("Head num: {}".format(classifier_head_num))
        logging.info("Learning rate: {}".format(classifier_learning_rate))
        logging.info("Diversity weight: {}".format(classifier_diversity_weight))
        logging.info("L2 reg weight: {}".format(classifier_l2_reg_weight))
        logging.info("Classifier train epochs: {}".format(classifier_train_epochs))

    
    def save(self):
        self.classifier.save(path=self.save_dir)
    
    def load(self):
        self.classifier.load(path=self.save_dir)

    
    def add_datafiles(self,
                      positive_files,
                      negative_files,
                      unlabelled_files):
        
        self.classifier.add_data(positive_files=positive_files,
                                 negative_files=negative_files,
                                 unlabelled_files=unlabelled_files)
    
    def train_classifier(self):
        self.classifier.set_class_weights()
        self.classifier.train(epochs=self.train_epochs, progress_bar=True)
    
    def test_classifier(self,
                        test_positive_files,
                        test_negative_files):
        dataset_positive = UnbalancedSetDataset(max_size=1e6,
                                                batchsize=64)
        
        dataset_negative = UnbalancedSetDataset(max_size=1e6,
                                                batchsize=64)
        
        dataset_positive.set_transform_function(transform)
        dataset_negative.set_transform_function(transform)
        dataset_positive.set_load_data_function(load_image)
        dataset_negative.set_load_data_function(load_image)
        
        
        dataset_positive.add_true_files(test_positive_files)
        dataset_negative.add_false_files(test_negative_files)
        
        counter = 0
        accuracy = np.zeros(self.classifier.head_num)
        accuracy_pos = np.zeros(self.classifier.head_num)
        accuracy_neg = np.zeros(self.classifier.head_num)
        
        for _ in range(dataset_positive.num_batches):
            counter += 1
            x, y = dataset_positive.get_batch()
            pred_y, votes = self.classifier.predict(x)

            pred_y = pred_y.cpu()
            
            for idx in range(self.classifier.head_num):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach()
                accuracy_pos[idx] += (torch.sum(pred_class==y).item())/len(y)
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)
        
        accuracy_pos /= counter
        
        total_count = counter
        counter = 0
        
        for _ in range(dataset_negative.num_batches):
            counter += 1
            x, y = dataset_negative.get_batch()
            pred_y, votes = self.classifier.predict(x)

            pred_y = pred_y.cpu()
            
            for idx in range(self.classifier.head_num):
                pred_class = torch.argmax(pred_y[:,idx,:], dim=1).detach()
                accuracy_neg[idx] += (torch.sum(pred_class==y).item())/len(y)
                accuracy[idx] += (torch.sum(pred_class==y).item())/len(y)
        
        accuracy_neg /= counter
        total_count += counter
        
        accuracy /= total_count
        
        weighted_acc = (accuracy_pos + accuracy_neg)/2
        
        logging.info("============= Classifiers evaluated =============")
        for idx in range(self.classifier.head_num):
            logging.info("Head idx:{:<4}, True accuracy: {:.4f}, False accuracy: {:.4f}, Total accuracy: {:.4f}, Weighted accuracy: {:.4f}".format(
                idx,
                accuracy_pos[idx],
                accuracy_neg[idx],
                accuracy[idx],
                weighted_acc[idx])
            )
        logging.info("=================================================")
        
        return accuracy, weighted_acc
    

    
    