import logging 
import datetime 
import os 
import random 
import gin 
import torch 
import lzma 
import dill 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from portable.utils.utils import set_seed
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from portable.option import AttentionOption
from portable.option.ensemble.custom_attention import AutoEncoder

from portable.agent.option_agent import OptionAgent

@gin.configurable
class AdvancedMinigridExperiment():
    def __init__(self,
                 base_dir,
                 experiment_name,
                 training_seed,
                 experiment_seed,
                 action_agent,
                 option_agent,
                 global_option,
                 num_options,
                 markov_option_builder,
                 policy_phi,
                 num_primitive_actions=7,
                 dataset_transform_function=None,
                 initiation_epochs=300,
                 termination_epochs=300,
                 policy_lr=1e-4,
                 policy_max_steps=1e6,
                 policy_success_threshold=0.98,
                 use_gpu=True,
                 names=None):
        
        
        self.training_seed = training_seed
        self.initiation_epochs=initiation_epochs
        self.termination_epochs=termination_epochs
        self.policy_lr=policy_lr
        self.policy_max_steps=policy_max_steps
        self.num_options = num_options
        self.policy_success_threshold = policy_success_threshold
        self.use_gpu = use_gpu
        self.num_primitive_actions = num_primitive_actions
        
        "FIRST OPTION IS GLOBAL OPTION gives access to primitive skills"
        #################################################
        #                     TODO                      #
        #################################################
        #   Change policy bootstrap to only consider    # 
        #                 one policy                    #
        #################################################
        
        if self.use_gpu:
            self.embedding = AutoEncoder().to("cuda")
        else:
            self.embedding = AutoEncoder()
        self.embedding_loaded = False
        
        set_seed(experiment_seed)
        self.seed = experiment_seed
        self.name = experiment_name
        self.base_dir = os.path.join(base_dir, experiment_name, str(experiment_seed))
        self.log_dir = os.path.join(self.base_dir, 'logs')
        self.plot_dir = os.path.join(self.base_dir, 'plots')
        self.save_dir = os.path.join(self.base_dir, 'checkpoints')
        
        self.buffer = deque(maxlen=200)
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)      
        log_file = os.path.join(self.log_dir, "{}.log".format(datetime.datetime.now()))
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s %(levelname)s: %(message)s',
                            level=logging.INFO)
        logging.info("[experiment] Beginning experiment {} seed {}".format(self.name, self.seed))
        logging.info("======== HYPERPARAMETERS ========")
        logging.info("Experiment seed: {}".format(experiment_seed))
        logging.info("Training seed: {}".format(training_seed))
        
        self.trial_data = []
        
        self.agent = OptionAgent(action_agent=action_agent,
                                 option_agent=option_agent,
                                 use_gpu=use_gpu,
                                 phi=policy_phi,
                                 summary_writer=self.writer)
        
        option_save_dirs = os.path.join(self.save_dir, 'options')
        
        self.options = [AttentionOption(use_gpu=use_gpu,
                                        log_dir=os.path.join(self.log_dir, 'option_'+str(x)),
                                        markov_option_builder=markov_option_builder,
                                        embedding=self.embedding,
                                        policy_phi=policy_phi,
                                        num_actions= num_primitive_actions,
                                        dataset_transform_function=dataset_transform_function,
                                        save_dir=os.path.join(option_save_dirs, str(x)),
                                        option_name=names[x] if names is not None else None) for x in range(num_options)]

        self.global_option = global_option
    
    def save(self):
        self.agent.save(self.save_dir)
        filename = os.path.join(self.save_dir, 'experiment_data.pkl')
        with lzma.open(filename, 'wb') as f:
            dill.dump(self.trial_data, f)
        
        for idx in range(self.num_options):
            self.options[idx].save()

    def load(self):
        self.agent.load(self.save_dir)
        filename = os.path.join(self.save_dir, 'experiment_data.pkl')
        if os.path.exists(filename):
            with lzma.open(filename, 'rb') as f:
                self.trial_data = dill.load(f)
        for idx in range(self.num_options):
            self.options[idx].load()


    def load_embedding(self, load_dir=None):
        if load_dir is None:
            load_dir = os.path.join(self.save_dir, 'embedding', 'model.ckpt')
        logging.info("[experiment embedding] Embedding loaded from {}".format(load_dir))
        self.embedding.load_state_dict(torch.load(load_dir))
        self.embedding_loaded = True
    
    def train_embedding(self, 
                        train_data,
                        epochs,
                        lr):
        optimizer = torch.optim.Adam(self.embedding.parameters(), lr=lr)
        mse_loss = torch.nn.MSELoss()
        base_dir = os.path.join(self.save_dir, 'embedding')
        os.makedirs(base_dir, exist_ok=True)
        save_file = os.path.join(self.save_dir, 'embedding', 'model.ckpt')
        train_x = None
        
        for epoch in range(epochs):
            train_data.shuffle()
            loss = 0
            counter_train = 0
            logging.info("[experiment embedding] Training embedding")
            for b_idx in range(train_data.num_batches):
                counter_train += 1
                x, _ = train_data.get_batch()
                x = x.to(self.device)
                train_x = x[:5]
                pred = self.embedding(x)
                mse_loss = mse_loss(pred, x)
                loss += mse_loss.item()
                mse_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            self.writer.add_scalar('embedding/mse_loss', loss/counter_train, epoch)
            for i in range(5):
                fig, axes = plt.subplots(ncols=2)
                sample = train_x[i]
                with torch.no_grad():
                    axes[0].set_axis_off()
                    axes[1].set_axis_off()
                    with torch.no_grad():
                        pred = self.embedding(sample.unsqueeze(0)).cpu().numpy()
                    axes[0].imshow(np.transpose(sample.cpu().numpy(), axes=(1,2,0)))
                    axes[1].imshow(np.transpose(pred[0], axes=(1,2,0)))

                    fig.savefig(os.path.join(base_dir, "{}.png".format(i)))
                plt.close(fig)
        torch.save(self.embedding.state_dict(), save_file)
        self.embedding_loaded = True
    
    def add_datafiles(self,
                      initiation_positive_files,
                      initiation_negative_files,
                      termination_positive_files,
                      termination_negative_files):
        assert len(initiation_positive_files) == self.num_options
        assert len(initiation_negative_files) == self.num_options
        assert len(termination_positive_files) == self.num_options
        assert len(termination_negative_files) == self.num_options
        
        for idx in range(self.num_options):
            self.options[idx].initiation.add_data_from_files(initiation_positive_files[idx],
                                                             initiation_negative_files[idx])
            self.options[idx].termination.add_data_from_files(termination_positive_files[idx],
                                                              termination_negative_files[idx])
    
    def train_options(self,
                      training_envs):
        # train all options
        for idx in range(self.num_options):
            logging.info('[experiment] Training option {}'.format(idx))
            self.options[idx].initiation.train(self.initiation_epochs)
            self.options[idx].termination.train(self.termination_epochs)
            self.options[idx].bootstrap_policy(training_envs[idx],
                                               self.policy_max_steps,
                                               self.policy_success_threshold)
            
            self.options[idx].save()
    
    def run_episode(self,
                    env):
        # run a single experiment
        obs, info = env.reset()
        done = False
        episode_reward = 0
        rewards = []
        steps = 0
        
        while not done:
            self.add_state_to_buffer(obs)
            next_obs, reward, done, action, option, info, step_num = self.run_one_step(env, obs, info)
            rewards.append(reward)
            
            self.agent.observe(obs=obs,
                               action=action,
                               option_idx=option,
                               rewards=reward,
                               next_obs=next_obs,
                               terminal=done)
            
            obs = next_obs
            episode_reward += sum(reward)
            steps += step_num
        
        print("[rainbow agent] steps: {} undiscounter average reward: {}".format(steps,
                                                                                 np.sum(rewards)))
        return rewards, steps
    
    def add_state_to_buffer(self, state):
        if type(state) is np.ndarray:
            state = torch.from_numpy(state).float()
        if torch.max(state) > 1:
            state = state/255.0
        
        self.buffer.append(state)
    
    def run_one_step(self, env, state, info):
        action_mask = [1]
        option_mask = [[1*self.num_primitive_actions]]
        
        for option in self.options:
            can_execute, available_options = option.can_initiate(state, info)
            action_mask.append(can_execute)
            option_mask.append(available_options)
        
        action, option_idx = self.agent.act(state,
                                        action_mask,
                                        option_mask)
        
        if action == 0:
            action = self.global_option.act(state)
            next_obs, reward, done, info = env.step(action)

            return next_obs, [reward], done, 0, 0, info, 1
        else:
            action_idx = action - 1
            
            if len(self.buffer) >= 20:
                false_states = random.sample(list(self.buffer), 20)
            else:
                false_states = list(self.buffer)
            
            next_obs, reward, done, info, steps = self.options[action_idx].run(env,
                                                                               state,
                                                                               info,
                                                                               option_idx=option_idx,
                                                                               eval=False,
                                                                               false_states=false_states)
            return next_obs, reward, done, action, option, info, steps
            
    def run(self,
            make_env,
            num_envs,
            frames_per_env):
        # run experiment
        # if self.embedding_loaded is not True:
        #     raise Exception("Embedding has not yet been loaded")
        
        logging.info("[experiment] Beginning experiment")
        logging.info("[experiment] Training on {} environments, {} frames per environment".format(num_envs,
                                                                                                  frames_per_env))
        
        for idx in range(num_envs):
            seed = random.randint(0, 1000)
            while seed == self.training_seed:
                seed = random.randint(0, 1000)
            
            env = make_env(seed)
            frames = 0
            while frames < frames_per_env:
                episode_rewards, steps = self.run_episode(env)
                undiscounted_return = sum(episode_rewards)
                frames += steps 
                self.writer.add_scalar('undiscounter_return',
                                       undiscounted_return,
                                       frames)
                
                self.trial_data.append({"reward": episode_rewards,
                                        "frames":frames,
                                        "seed": seed,
                                        "env_idx": idx})
                
                print(100*'-')
                print(f'Env seed: {str(seed)}',
                f"Env Frames': {str(frames)}",
                f'Reward: {str(undiscounted_return)}')
                print(100 * '-')
                
                logging.info('[experiment] Env seed: {}'.format(seed))
                logging.info("Env Frames': {}".format(frames))
                logging.info('Reward: {}'.format(undiscounted_return))
                logging.info(100 * '-')