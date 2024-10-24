import argparse
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm
#import torch_tensorrt

from portable.utils.utils import load_gin_configs
from experiments.classifier.core.classifier_experiment import DivDisClassifierExperiment

dataset_path = 'resources/jaco/handle'

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

    #torch.set_float32_matmul_precision('high')
    #torch.backends.cuda.matmul.allow_tf32 = True
    #torch.set_float32_matmul_precision('medium')
    #torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    #print(torch._dynamo.list_backends())
    seeds = [args.seed + 20*i for i in range(args.n)]

    best_total_acc = []
    best_weighted_acc = []
    avg_weighted_acc = []
    train_time = []
    test_time = []

    for i in tqdm(range(args.n)):
        cur_seed = seeds[i]
            
        t0 = time.time()
        experiment = DivDisClassifierExperiment(
                            base_dir=args.base_dir,
                            seed=cur_seed)

        experiment.create_task_dict(dataset_path)
        tasks = list(experiment.task_dict.keys())
        
        train_tasks = ['microwave','recycle','stove','fridge']
        for t in train_tasks:
            tasks.remove(t)
        test_tasks = tasks

        train_positive_files, train_negative_files,_,_ = experiment.get_data_path(dataset_path, train_tasks, 'term')
        test_positive_files, test_negative_files,_,_ = experiment.get_data_path(dataset_path, test_tasks, 'term')
        unlabelled_train_files,_,_,_ = experiment.get_data_path(dataset_path, tasks, 'term')


        print(f"Train Positive Files: {len(train_positive_files)}")
        print(f"Train Negative Files: {len(train_negative_files)}")
        print(f"Unlabelled Train Files: {len(unlabelled_train_files)}")
        print(f"Test Positive Files: {len(test_positive_files)}")
        print(f"Test Negative Files: {len(test_negative_files)}")
        
        experiment.add_datafiles(train_positive_files,
                    train_negative_files,
                    unlabelled_train_files)

        experiment.train_classifier()
        
        t1 = time.time()
        train_time.append(t1-t0)
        t2 = time.time()
        accuracy = experiment.test_classifier(test_positive_files, test_negative_files)
        test_time.append(time.time()-t2)
        
        print(f"Total Accuracy: {np.round(accuracy[0], 4)}")
        print(f"Weighted Accuracy: {np.round(accuracy[1], 4)}")
        best_head = np.argmax(accuracy[1])
        best_weighted_acc.append(accuracy[1][best_head])
        best_total_acc.append(accuracy[0][best_head])
        avg_weighted_acc.append(np.mean(accuracy[1]))

    print(f"Best Total Accuracy: {np.mean(best_total_acc):.2f}, {np.std(best_total_acc):.2f}")
    print(f"Best Weighted Accuracy: {np.mean(best_weighted_acc):.2f}, {np.std(best_weighted_acc):.2f}")
    print(f"Avg Weighted Accuracy: {np.mean(avg_weighted_acc):.2f}, {np.std(avg_weighted_acc):.2f}")
    print(f"Train Time: {np.mean(train_time):.2f}, {np.std(train_time):.2f}")
    print(f"Test Time: {np.mean(test_time):.2f}, {np.std(test_time):.2f}")

    