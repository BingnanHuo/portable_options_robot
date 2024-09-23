from experiments.core.divdis_meta_masked_ppo_experiment import DivDisMetaMaskedPPOExperiment
import argparse
from portable.utils.utils import load_gin_configs
import torch 
from experiments.minigrid.utils import environment_builder
from experiments.minigrid.advanced_doorkey.core.policy_train_wrapper import AdvancedDoorKeyPolicyTrainWrapper
import random
from portable.agent.model.ppo import create_cnn_policy, create_cnn_vf
from experiments.divdis_minigrid.experiment_files import *
import numpy as np
from PIL import Image



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config_file", nargs='+', type=str, required=True)
    parser.add_argument("--gin_bindings", default=[], help='Gin bindings to override the values' + 
            ' set in the config files (e.g. "DQNAgent.epsilon_train=0.1",' +
            ' "create_atari_environment.game_name="Pong"").')
    
    args = parser.parse_args()
    load_gin_configs(args.config_file, args.gin_bindings)
    
    def policy_phi(x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = (x/255.0).float()
        return x
    
    def option_agent_phi(x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = (x/255.0).float()
        return x

    def termination_phi(x):
        img = Image.fromarray(x.numpy())
        return np.asarray(img.resize((128,128), Image.BICUBIC))/255
    
    experiment = DivDisMetaMaskedPPOExperiment(base_dir=args.base_dir,
                                               seed=args.seed,
                                               option_policy_phi=policy_phi,
                                               agent_phi=option_agent_phi,
                                               termination_phi=termination_phi,
                                               action_policy=create_cnn_policy(3,20),
                                               action_vf=create_cnn_vf(3),
                                               option_type="divdis")
    
    experiment.add_datafiles(minigrid_big_positive_files,
                             minigrid_big_negative_files,
                             minigrid_big_unlabelled_files)
    
    experiment.train_option_classifiers()
    
    experiment.test_classifiers(minigrid_big_test_files_positive,
                                minigrid_big_test_files_negative)
    
    
    meta_env = AdvancedDoorKeyPolicyTrainWrapper(environment_builder('SmallAdvancedDoorKey-16x16-v0',
                                                                     seed=args.seed,
                                                                     max_steps=int(15000),
                                                                     grayscale=False,
                                                                     scale_obs=True,
                                                                     final_image_size=(84,84),
                                                                     normalize_obs=False),
                                                 key_collected=False,
                                                 door_unlocked=False,
                                                 force_door_closed=True)
    
    experiment.train_meta_agent(meta_env,
                                args.seed,
                                4e6,
                                0.98)
    
    
