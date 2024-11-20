import datetime
import json
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
    # Convert to float and scale to [0.0, 1.0] range
    #x = x.float() / 255.0
    # Normalize using the ImageNet mean and std
    pipeline = transforms.Compose([
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                     std=[0.229, 0.224, 0.225])
    ])
    return pipeline(x)

def load_image(file_path):
    pipeline = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224), 
    ])
    image = Image.open(file_path).convert('RGB')
    image = pipeline(image)
    image_tensor = torch.tensor(np.array(image), dtype=torch.uint8).permute(2, 0, 1)  # Shape (C, H, W) in uint8
    return image_tensor.unsqueeze(0)

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

        self.task_dict = {}
        
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


    def create_task_dict(self, dataset_path):
        folders = os.listdir(dataset_path)
        folders.sort()
        tasks = [(folder.split('_')[1], folder.split('_')[2]) for folder in folders if len(folder.split('_')) == 3]
        for task_number, task_name in tasks:
            if task_name not in self.task_dict:
                self.task_dict[task_name] = []
            self.task_dict[task_name].append(task_number)
        return self.task_dict

    def get_data_path(self, dataset_path, task_names, init_term):
        if not isinstance(task_names, list):
            task_names = [task_names]

        matching_folders = []
        for task_name in task_names:
            for task_number in self.task_dict.get(task_name, []):
                folder_name = f"run_{task_number}_{task_name}"
                folder_path = os.path.join(dataset_path, folder_name, init_term)
                if os.path.exists(folder_path):
                    matching_folders.append(folder_path)
        
        positive_data_paths = []
        positive_labels = []
        negative_data_paths = []
        negative_labels = []

        for folder in matching_folders:
            pos_folder = os.path.join(folder, 'positive')
            neg_folder = os.path.join(folder, 'negative')

            # Load positive images and labels
            pos_images = sorted(os.listdir(pos_folder))
            with open(os.path.join(pos_folder, 'labels.json'), 'r') as f:
                pos_labels = json.load(f)
            for img in pos_images:
                if img in pos_labels:
                    positive_data_paths.append(os.path.join(pos_folder, img))
                    positive_labels.append(pos_labels[img])

            # Load negative images and labels
            neg_images = sorted(os.listdir(neg_folder))
            with open(os.path.join(neg_folder, 'labels.json'), 'r') as f:
                neg_labels = json.load(f)
            for img in neg_images:
                if img in neg_labels:
                    negative_data_paths.append(os.path.join(neg_folder, img))
                    negative_labels.append(neg_labels[img])

        return positive_data_paths, negative_data_paths, positive_labels, negative_labels

    
    def add_datafiles(self,
                      positive_files,
                      negative_files,
                      unlabelled_files):
        
        self.classifier.add_data(positive_files=positive_files,
                                 negative_files=negative_files,
                                 unlabelled_files=unlabelled_files)
    
    def train_classifier(self):
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
    

    
    