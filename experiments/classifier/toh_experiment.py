import argparse
import json
import os
import random
import stat
import time
import warnings

import numpy as np
import glob
import torch
from tqdm import tqdm
import transformers
#import torch_tensorrt

from portable.utils.utils import load_gin_configs
from experiments.classifier.core.classifier_experiment import DivDisClassifierExperiment
from experiments.classifier.core.utils import create_task_dict, get_data_path, filter_valid_images


def load_data(file_path):
        data = np.load(file_path, allow_pickle=True)
        data = torch.from_numpy(data)
        #print(f"Loaded data from {file_path} with shape {data.shape}")
        #if torch.max(data) <= 1 and torch.min(data) >= 0:
        #    data = data*255
        #data = data.to(torch.uint8)
        #data = data.squeeze()
        return data.unsqueeze(0)


def label_data(file_path, object_of_interest, skill_of_interest):
    # load json file
    with open(file_path, 'r') as f:
        json_dict = json.load(f)
    state_dict = json_dict['object_states'] # nested dict [stick: dict for levels (level: objects)]

    #check if the object of interest is at the right stick and level, for the skill of interest
    #obj_at_skill_term_loc = state_dict[skill_of_interest[0]][str(skill_of_interest[1])]
    objs_at_skill_term_loc = state_dict[skill_of_interest[0]].values()
    
    #if obj_at_skill_term_loc == object_of_interest:
    if object_of_interest in objs_at_skill_term_loc:
        return 1
    else:
        return 0

def carry_in_progress(file_path):
    # load json file
    with open(file_path, 'r') as f:
        json_dict = json.load(f)
    state_dict = json_dict['object_states'] # nested dict [stick: dict for levels (level: objects)]
    if state_dict['carrying'] is None:
        return False
    else:
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
        ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
        ' "create_atari_environment.game_name="Pong"").')

    args = parser.parse_args()
    load_gin_configs(args.config_file, args.gin_bindings)

    warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn.modules.lazy')
    transformers.logging.set_verbosity_error()  # This will show only errors, not warnings
    

    #torch.set_float32_matmul_precision('high')
    #torch.backends.cuda.matmul.allow_tf32 = True
    #torch.set_float32_matmul_precision('medium')
    #torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    #print(torch._dynamo.list_backends())
    seeds = [args.seed + i for i in range(args.n)]

    best_total_acc = []
    best_weighted_acc = []
    avg_weighted_acc = []
    train_time = []
    test_time = []

    base_dataset_path = 'resources/toh'
    train_set_names = [f'test{i}_light_on' for i in [1,2,3,4,5,6,7,8]]
    unlabeled_set_names = [f'test{i}_light_on' for i in [9,10,11,12,13,14,15]]
    test_set_names = [f'test{i}_light_on' for i in [16,17,18,19,20]]

    object_of_interest = ['cylinder', 2, 'blue'] # (shape, size, color)
    skill_of_interest = ['middle'] # (stick, level)

    # all datasets (seeds) containg the object of interest goes in labeled data. One set (task / seed) is used for training, the rest for testing. 
    # all other data goes in unlabelled data.


    train_image_paths = []
    train_json_paths = []
    for i in range(len(train_set_names)):
        train_set_path = os.path.join(base_dataset_path, train_set_names[i])
        #test_image_paths += glob.glob(os.path.join(test_set_path, 'images', '*.jpg'))
        train_image_paths += glob.glob(os.path.join(train_set_path, 'images', '*.npy'))
        train_json_paths += glob.glob(os.path.join(train_set_path, 'images','*.json'))

    idx_to_remove = []
    for j in range(len(train_image_paths)):
        cur_img = train_image_paths[j]
        cur_json = train_json_paths[j]
        if carry_in_progress(cur_json):
            idx_to_remove.append(j)
    train_image_paths = [train_image_paths[i] for i in range(len(train_image_paths)) if i not in idx_to_remove]
    train_json_paths = [train_json_paths[i] for i in range(len(train_json_paths)) if i not in idx_to_remove]
        

    test_image_paths = []
    test_json_paths = []
    for i in range(len(test_set_names)):
        test_set_path = os.path.join(base_dataset_path, test_set_names[i])
        #test_image_paths += glob.glob(os.path.join(test_set_path, 'images', '*.jpg'))
        test_image_paths += glob.glob(os.path.join(test_set_path, 'images', '*.npy'))
        test_json_paths += glob.glob(os.path.join(test_set_path, 'images','*.json'))
    idx_to_remove = []
    for j in range(len(test_image_paths)):
        cur_img = test_image_paths[j]
        cur_json = test_json_paths[j]
        if carry_in_progress(cur_json):
            idx_to_remove.append(j)
    test_image_paths = [test_image_paths[i] for i in range(len(test_image_paths)) if i not in idx_to_remove]
    test_json_paths = [test_json_paths[i] for i in range(len(test_json_paths)) if i not in idx_to_remove]

            
    train_image_paths.sort()
    train_json_paths.sort()
    
    assert len(train_image_paths) == len(train_json_paths), f"Number of images and json files do not match in {train_set_path}"
    assert len(test_image_paths) == len(test_json_paths), f"Number of images and json files do not match in {test_set_path}"

    unlabeled_image_paths = []
    unlabeled_json_paths = []
    for i in range(len(unlabeled_set_names)):
        unlabeled_set_path = os.path.join(base_dataset_path, unlabeled_set_names[i])
        unlabeled_image_paths += glob.glob(os.path.join(unlabeled_set_path, 'images', '*.npy'))
        unlabeled_json_paths += glob.glob(os.path.join(unlabeled_set_path, 'images','*.json'))
    idx_to_remove = []
    for j in range(len(unlabeled_image_paths)):
        cur_img = unlabeled_image_paths[j]
        cur_json = unlabeled_json_paths[j]
        if carry_in_progress(cur_json):
            idx_to_remove.append(j)
    unlabeled_image_paths = [unlabeled_image_paths[i] for i in range(len(unlabeled_image_paths)) if i not in idx_to_remove]
    unlabeled_json_paths = [unlabeled_json_paths[i] for i in range(len(unlabeled_json_paths)) if i not in idx_to_remove]

    assert len(unlabeled_image_paths) == len(unlabeled_json_paths), f"Number of images and json files do not match in {unlabeled_set_path}"
    

    # put into positive and negative files
    train_positive_files = []
    train_negative_files = []
    for i in range(len(train_image_paths)):
        image_path = train_image_paths[i]
        json_path = train_json_paths[i]
        label = label_data(json_path, object_of_interest, skill_of_interest)
        if label == 1:
            train_positive_files.append(image_path)
        else:
            train_negative_files.append(image_path)
    assert len(train_positive_files) + len(train_negative_files) == len(train_image_paths), f"Number of positive and negative files do not match in {train_set_path}"

    test_positive_files = []
    test_negative_files = []
    for i in range(len(test_image_paths)):
        image_path = test_image_paths[i]
        json_path = test_json_paths[i]
        label = label_data(json_path, object_of_interest, skill_of_interest)
        if label == 1:
            test_positive_files.append(image_path)
        else:
            test_negative_files.append(image_path)
    assert len(test_positive_files) + len(test_negative_files) == len(test_image_paths), f"Number of positive and negative files do not match in test sets"

    unlabeled_files = unlabeled_image_paths

    # print length of each set
    print(f"Train positive set: {len(train_positive_files)}")
    print(f"Train negative set: {len(train_negative_files)}")
    print(f"Unlabeled set:      {len(unlabeled_files)}")
    print(f"Test positive set:  {len(test_positive_files)}")
    print(f"Test negative set:  {len(test_negative_files)}")
    


    for i in tqdm(range(args.n)):
        t0 = time.time()
        cur_seed = seeds[i]

        experiment = DivDisClassifierExperiment(
                            base_dir=args.base_dir,
                            seed=cur_seed,
                            experiment_name=f"{object_of_interest}_at_{skill_of_interest}",
                            load_data=load_data)
        experiment.add_datafiles(train_positive_files,
                    train_negative_files,
                    unlabeled_files)

        experiment.train_classifier()
        
        t1 = time.time()
        train_time.append(t1-t0)
        t2 = time.time()
        accuracy = experiment.test_classifier(test_positive_files, test_negative_files)
        test_time.append(time.time()-t2)
        
        print(f"Total Accuracy:    {np.round(accuracy[0], 2)}")
        print(f"Weighted Accuracy: {np.round(accuracy[1], 2)}")
        best_head = np.argmax(accuracy[1])
        best_weighted_acc.append(accuracy[1][best_head])
        best_total_acc.append(accuracy[0][best_head])
        avg_weighted_acc.append(np.mean(accuracy[1]))

    print(f"Best Total Accuracy:    {np.mean(best_total_acc):.2f}, {np.std(best_total_acc):.2f}")
    print(f"Best Weighted Accuracy: {np.mean(best_weighted_acc):.2f}, {np.std(best_weighted_acc):.2f}")
    print(f"Avg Weighted Accuracy:  {np.mean(avg_weighted_acc):.2f}, {np.std(avg_weighted_acc):.2f}")
    print(f"Train Time: {np.mean(train_time):.1f}, {np.std(train_time):.1f}")
    print(f"Test Time:  {np.mean(test_time):.1f}, {np.std(test_time):.1f}")


    print(f"Evaluating on unlabeled data")
    experiment.unlabeled_predictions(unlabeled_files, 300)