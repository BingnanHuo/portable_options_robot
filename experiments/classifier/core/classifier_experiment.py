import datetime
import logging
import os
import pickle
import shutil

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
        #transforms.RandomRotation(degrees=(180, 180)) # added for toh data
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
                 classifier_model_name,
                 transform=transform,
                 load_data=load_image):
        
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

        self.transform = transform
        self.load_data = load_data
        
        self.classifier.dataset.set_transform_function(self.transform)
        self.classifier.dataset.set_load_data_function(self.load_data)
        
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
        
        dataset_positive.set_transform_function(self.transform)
        dataset_negative.set_transform_function(self.transform)
        dataset_positive.set_load_data_function(self.load_data)
        dataset_negative.set_load_data_function(self.load_data)
        
        
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
    

    
    def unlabeled_predictions(self,
                              unlabeled_files,
                              num_samples=None):
        """
        Runs unlabeled data through each classifier head, categorizes predictions,
        and copies the images to corresponding output folders with renamed filenames.

        Args:
            unlabeled_files (list): A list of file paths for the unlabeled images.

            num_samples (int, optional): The maximum number of unlabeled files to process.
                                         If None, all files are processed. Defaults to None.

        Returns:
            list: A list of dictionaries, where each dictionary corresponds to a classifier head.
                  Each dictionary contains two keys: 'positive' and 'negative', with values
                  being lists of the *original* file paths classified accordingly by that head.
                  Example: [{'positive': [path1, path5], 'negative': [path2]}, {'positive': [...], 'negative': [...]}, ...]
        """
        
        output_base_dir = os.path.join(self.plot_dir, 'unlabeled_predictions')  

        batch_size = 64 # Define a suitable batch size for prediction
        results = [{'positive': [], 'negative': []} for _ in range(self.classifier.head_num)]
        processed_count = 0

        # --- Create output directories ---
        try:
            for head_idx in range(self.classifier.head_num):
                pos_dir = os.path.join(output_base_dir, f'head_{head_idx}', 'positive')
                neg_dir = os.path.join(output_base_dir, f'head_{head_idx}', 'negative')
                os.makedirs(pos_dir, exist_ok=True)
                os.makedirs(neg_dir, exist_ok=True)
            logging.info(f"[unlabeled_predictions] Created output directories under {output_base_dir}")
        except OSError as e:
            logging.error(f"[unlabeled_predictions] Error creating output directories: {e}")
            return None # Or raise the exception, depending on desired behavior
        # --- End directory creation ---

        files_to_process = unlabeled_files
        if num_samples is not None and num_samples > 0 and num_samples < len(files_to_process):
            # Optionally use random.sample for random subset:
            # import random
            # files_to_process = random.sample(unlabeled_files, num_samples)
            files_to_process = files_to_process[:num_samples]

        logging.info(f"[unlabeled_predictions] Starting prediction and copying for {len(files_to_process)} unlabeled files.")


        with torch.no_grad(): # Disable gradient calculations for inference
            for i in range(0, len(files_to_process), batch_size):
                batch_files = files_to_process[i:min(i + batch_size, len(files_to_process))]
                batch_x_list = []
                batch_files_loaded = [] # Keep track of files successfully loaded in this batch

                # Load images for the current batch
                for file_path in batch_files:
                    image_tensor = self.load_data(file_path)
                    if image_tensor is not None:
                        batch_x_list.append(image_tensor)
                        batch_files_loaded.append(file_path)
                    else:
                        logging.warning(f"[unlabeled_predictions] Skipping file due to loading error: {file_path}")

                if not batch_x_list:
                    logging.warning(f"[unlabeled_predictions] No images loaded in batch starting at index {i}. Skipping.")
                    continue

                batch_x = torch.cat(batch_x_list, dim=0)

                # Perform prediction
                pred_y, _ = self.classifier.predict(batch_x) # pred_y shape: [batch_size, head_num, num_classes]
                pred_y = pred_y.cpu() # Shape: [batch_size, head_num, num_classes]
                probabilities = torch.softmax(pred_y, dim=2) # Calculate probabilities along the class dimension

                # --- Define single output directory ---
                annotated_output_dir = os.path.join(self.plot_dir, 'unlabeled_predictions_annotated')
                try:
                    os.makedirs(annotated_output_dir, exist_ok=True)
                    if i == 0: # Log only once
                         logging.info(f"[unlabeled_predictions] Saving annotated images to {annotated_output_dir}")
                except OSError as e:
                    logging.error(f"[unlabeled_predictions] Error creating annotated output directory: {e}")
                    # Decide how to handle this - maybe return or raise
                    return None # Or raise e
                # --- End directory definition ---


                # Process predictions and save annotated images for each sample in the batch
                for sample_idx in range(len(batch_files_loaded)):
                    original_file_path = batch_files_loaded[sample_idx]

                    # --- Generate new filename (base_name.jpg) ---
                    try:
                        parent_dir_name = os.path.basename(os.path.dirname(original_file_path))
                        original_filename = os.path.basename(original_file_path)
                        base_name, _ = os.path.splitext(original_filename) # Get base name without extension
                        new_filename = f"{parent_dir_name}_{base_name}.jpg" # Force .jpg extension
                        destination_path = os.path.join(annotated_output_dir, new_filename)
                    except Exception as e:
                         logging.error(f"Error generating new filename for {original_file_path}: {e}")
                         continue # Skip this file if naming fails
                    # --- End filename generation ---

                    # --- Create annotation text ---
                    annotation_lines = []
                    for head_idx in range(self.classifier.head_num):
                        # Assuming class 1 is 'positive' and class 0 is 'negative'
                        prob_positive = probabilities[sample_idx, head_idx, 1].item() # Get probability of class 1
                        pred_class = torch.argmax(pred_y[sample_idx, head_idx, :]).item()
                        annotation_lines.append(f"Head {head_idx}: Pred={pred_class}, P(pos)={prob_positive:.3f}")
                    annotation_text = "\n".join(annotation_lines)
                    # --- End annotation text ---

                    # --- Load, annotate, and save image using matplotlib ---
                    try:
                        # Load the original image again for plotting
                        # Use PIL to ensure consistent loading as used in load_data
                        img = Image.open(original_file_path).convert('RGB')

                        fig, ax = plt.subplots(figsize=(8, 8)) # Adjust figsize as needed
                        ax.imshow(img)
                        ax.set_title(annotation_text, fontsize=8) # Add predictions as title
                        ax.axis('off') # Hide axes

                        plt.savefig(destination_path, bbox_inches='tight', pad_inches=0.1)
                        plt.close(fig) # Close the figure to free memory

                    except Exception as e:
                        logging.error(f"Error processing or saving annotated image for {original_file_path} to {destination_path}: {e}")
                    # --- End image annotation and saving ---


                processed_count += len(batch_files_loaded)
                if processed_count % (batch_size * 5) == 0 or processed_count == len(files_to_process):
                     logging.info(f"[unlabeled_predictions] Processed and saved annotations for {processed_count}/{len(files_to_process)} files...")


        logging.info("============= Unlabeled Data Annotation Summary =============")
        logging.info(f"Annotated images saved in: {annotated_output_dir}")
        logging.info("============================================================")

        # No longer returning categorized lists
        return None # Or return the path to the output directory