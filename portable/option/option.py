from collections import deque
import logging
import os
import pickle
import numpy as np
import gin
from portable.option.policy.agents import evaluating

from portable.option.sets import Set
from portable.option.sets.utils import PositionSetPair
from portable.option.policy.agents import EnsembleAgent
from portable.utils.utils import plot_state

logger = logging.getLogger(__name__)
import time

import matplotlib.pyplot as plt

@gin.configurable
class Option():
    def __init__(
            self,
            device,

            initiation_vote_function,
            termination_vote_function,
            policy_phi,
            action_selection_strategy,
            prioritized_replay_anneal_steps,

            policy_warmup_steps=500000,
            policy_batchsize=32,
            policy_buffer_length=100000,
            policy_update_interval=4,
            q_target_update_interval=40,
            policy_embedding_output_size=64,
            policy_learning_rate=2.5e-4,
            final_epsilon=0.01,
            final_exploration_frames=10**6,
            discount_rate=0.9,
            policy_attention_module_num=8,
            policy_num_output_classes=18,

            initiation_beta_distribution_alpha=30,
            initiation_beta_distribution_beta=5,
            initiation_attention_module_num=8,
            initiation_embedding_learning_rate=1e-4,
            initiation_classifier_learning_rate=1e-2,
            initiation_embedding_output_size=64,
            initiation_dataset_max_size=50000,

            termination_beta_distribution_alpha=30,
            termination_beta_distribution_beta=5,
            termination_attention_module_num=8,
            termination_embedding_learning_rate=1e-4,
            termination_classifier_learning_rate=1e-2,
            termination_embedding_output_size=64,
            termination_dataset_max_size=50000,

            markov_termination_epsilon=3,
            min_interactions=100,
            timeout=50,
            allowed_additional_loss=2,

            log=True):
        
        self.policy = EnsembleAgent(
            device=device,
            warmup_steps=policy_warmup_steps,
            batch_size=policy_batchsize,
            phi=policy_phi,
            action_selection_strategy=action_selection_strategy,
            prioritized_replay_anneal_steps=prioritized_replay_anneal_steps,
            buffer_length=policy_buffer_length,
            update_interval=policy_update_interval,
            q_target_update_interval=q_target_update_interval,
            embedding_output_size=policy_embedding_output_size,
            learning_rate=policy_learning_rate,
            final_epsilon=final_epsilon,
            final_exploration_frames=final_exploration_frames,
            discount_rate=discount_rate,
            num_modules=policy_attention_module_num,
            num_output_classes=policy_num_output_classes
        )

        self.initiation = Set(
            device=device,
            vote_function=initiation_vote_function,
            beta_distribution_alpha=initiation_beta_distribution_alpha,
            beta_distribution_beta=initiation_beta_distribution_beta,
            attention_module_num=initiation_attention_module_num,
            embedding_learning_rate=initiation_embedding_learning_rate,
            classifier_learning_rate=initiation_classifier_learning_rate,
            embedding_output_size=initiation_embedding_output_size,
            dataset_max_size=initiation_dataset_max_size
        )

        self.termination = Set(
            device=device,
            vote_function=termination_vote_function,
            beta_distribution_alpha=termination_beta_distribution_alpha,
            beta_distribution_beta=termination_beta_distribution_beta,
            attention_module_num=termination_attention_module_num,
            embedding_learning_rate=termination_embedding_learning_rate,
            classifier_learning_rate=termination_classifier_learning_rate,
            embedding_output_size=termination_embedding_output_size,
            dataset_max_size=termination_dataset_max_size
        )

        self.markov_classifiers = []
        self.markov_termination_epsilon = markov_termination_epsilon
        self.min_interactions = min_interactions
        self.option_timeout = timeout
        
        self.markov_classifier_idx = None
        self.use_log = log
        self.allowed_loss = allowed_additional_loss

        
    def log(self, message):
        if self.use_log:
            logger.info(message)

    @staticmethod
    def _get_save_paths(path):
        policy = os.path.join(path, 'policy')
        initiation = os.path.join(path, 'initiation')
        termination = os.path.join(path, 'termination')

        return policy, initiation, termination

    def save(self, path):
        policy_path, initiation_path, termination_path = self._get_save_paths(path)
        
        os.makedirs(policy_path, exist_ok=True)
        os.makedirs(initiation_path, exist_ok=True)
        os.makedirs(termination_path, exist_ok=True)
        
        self.policy.save(policy_path)
        self.initiation.save(initiation_path)
        self.termination.save(termination_path)

        self.log("[option] Saving option to path: {}".format(path))

    def load(self, path):
        policy_path, initiation_path, termination_path = self._get_save_paths(path)

        if os.path.exists(os.path.join(policy_path, 'agent.pkl')):
            self.policy = self.policy.load(os.path.join(policy_path, 'agent.pkl'))
        self.initiation.load(initiation_path)
        self.termination.load(termination_path)

        self.log("[option] Loading option from path: {}".format(path))

    def _add_markov_classifier(
            self,
            markov_states,
            positions,
            termination):
        # adds a markov classifier from the previous run of the option
        self.markov_classifiers.append(
            PositionSetPair(
                markov_states,
                positions,
                termination,
                self.markov_termination_epsilon
            )
        )

        self.log("[option] Added markov classifier")

    def can_initiate(self, agent_space_state, markov_space_state):
        # check if option can initiate
        vote_global = self.initiation.vote(agent_space_state)
        vote_markov = False
        self.markov_classifier_idx = None
        for idx in range(len(self.markov_classifiers)):
            prediction = self.markov_classifiers[idx].can_initiate(markov_space_state)
            if prediction == 1:
                self.markov_classifier_idx = idx
                vote_markov = True
        
        return vote_global, vote_markov

    def run(self, 
            env, 
            state, 
            info,
            use_global_classifiers=True):
        # run the option. Policy takes over control
        steps = 0
        total_reward = 0
        agent_space_states = []
        agent_state = info["stacked_agent_state"]
        positions = []

        while steps < self.option_timeout:
            #double check this, don't really wanna save whole info
            agent_space_states.append(agent_state)
            positions.append((info["player_x"], info["player_y"]))
            
            action = self.policy.act(state)

            next_state, reward, done, info = env.step(action)
            # screen = env.render('rgb_array')
            # fig = plt.figure(num=1, clear=True)
            # ax = fig.add_subplot()
            # ax.imshow(screen)
            # ax.axis('off')
            # time.sleep(0.5)
            # plt.show(block=False)
            agent_state = info["stacked_agent_state"]
            steps += 1

            if use_global_classifiers:
                should_terminate = self.termination.vote(agent_state)
            else:
                should_terminate = self.markov_classifiers[self.markov_classifier_idx].can_terminate((info["player_x"],info["player_y"]))
            
            self.policy.observe(state, action, reward, next_state, done)
            total_reward += reward

            # need death condition => maybe just terminate on life lost wrapper
            if done or info['needs_reset'] or should_terminate:
                # ended episode without dying => success
                # detected the end of the option => success
                if (done or should_terminate) and not info['dead']:
                    self._option_success(
                        positions,
                        agent_space_states,
                        info,
                        agent_state
                    )
                    self.log("[option run] Option completed successfully (done: {}, should_terminate: {})".format(done, should_terminate))
                    info['option_timed_out'] = False
                    return next_state, total_reward, done, info, steps
                
                # episode timed out => not a fail or success 
                # no failure or success
                if info['needs_reset']:
                    self.log("[option run] Environment needs reset. Returning")
                    info['option_timed_out'] = False
                    return next_state, total_reward, done, info, steps 

                # we died during option execution => fail
                if done and info['dead']:
                    self.log("[option run] Option failed because agent died. Returning \n\t {}".format(info['position']))
                    self._option_fail(
                        positions,
                        agent_space_states
                    )
                    info['option_timed_out'] = False
                    return next_state, total_reward, done, info, steps
            
            state = next_state

        # we didn't find a valid termination for the option before the 
        # allowed execution time ran out => fail
        self._option_fail(
            positions,
            agent_space_states
        )
        self.log("[option] Option timed out. Returning\n\t {}".format(info['position']))
        info['option_timed_out'] = True

        return next_state, total_reward, done, info, steps
            
    def evaluate(self,
                env,
                state,
                info,
                use_global_classifiers=True):
        # runs option in evaluation mode (does not store transitions for later training)
        # does not create Markov classifiers
        steps = 0
        total_reward = 0
        with evaluating(self.policy):
            while steps < self.option_timeout:
                action = self.policy.act(state)

                next_state, reward, done, info = env.step(action)
                # screen = env.render('rgb_array')
                # fig = plt.figure(num=1, clear=True)
                # ax = fig.add_subplot()
                # ax.imshow(screen)
                # ax.axis('off')
                # # time.sleep(0.5)
                # plt.show(block=False)
                # input(info["position"])
                agent_state = info["stacked_agent_state"]
                steps += 1

                if use_global_classifiers:
                    should_terminate = self.termination.vote(agent_state)
                else:
                    should_terminate = self.markov_classifiers[self.markov_classifier_idx].can_terminate((info["player_x"],info["player_y"]))

                self.policy.observe(state, action, reward, next_state, done)
                total_reward += reward

                if done or info['needs_reset'] or should_terminate:
                    if (done or should_terminate) and not info['dead']:
                        self.log("[option eval] Option completed successfully (done: {}, should_terminate {})".format(done, should_terminate))
                        info['option_timed_out'] = False
                        return next_state, total_reward, done, info, steps
                        
                    if info['needs_reset']:
                        self.log("[option eval] Environment needs reset. Returning")
                        info['option_timed_out'] = False
                        return next_state, total_reward, done, info, steps

                    if done and info['dead']:
                        self.log("[option eval] Option failed because agent died. Returning")
                        info['option_timed_out'] = False
                        return next_state, total_reward, done, info, steps
                
                state = next_state

        self.log("[option eval] Option timed out. Returning")
        info['option_timed_out'] = True

        return next_state, total_reward, done, info, steps

    def _test_markov_classifier(
            self,
            initiation_states,
            termination_states,
            test_negative_only=False):

        if test_negative_only:
            # only test that we don't add too much loss if is negative
            self.log("[option] Testing negative samples only from Markov classifier")
            initiation_loss = self.initiation.loss([], initiation_states)
            if np.mean(self.initiation.confidence.weights*initiation_loss) > self.allowed_loss*np.mean(self.initiation.confidence.weights*self.initiation.avg_loss):
                # too much loss added to initiation classifier
                self.log("[option] Initiation loss too high. Data not added")
                return False
            else:
                return True

        self.log("[option] Testing samples from Markov classifier")

        # check if we can add data from this markov classifier
        if self.markov_classifiers[self.markov_classifier_idx].interaction_count < self.min_interactions:
            # we have not interacted with this markov classifier enough to trust it
            self.log("[option] Not enough interactions with Markov classifier. Data not added.")
            return False
        
        initiation_loss = self.initiation.loss(
            initiation_states,
            termination_states
        )
        termination_loss = self.termination.loss(
            termination_states,
            initiation_states
        )

        if np.mean(self.initiation.confidence.weights*initiation_loss) > self.allowed_loss*np.mean(self.initiation.confidence.weights*self.initiation.avg_loss):
            # too much loss added to initiation classifier
            self.log("[option] Data loss: {} avg loss: {}".format(self.initiation.confidence.weights*initiation_loss, self.initiation.confidence.weights*self.initiation.avg_loss))
            self.log("[option] Initiation loss too high. Data not added")
            return False
        
        if np.mean(self.termination.confidence.weights*termination_loss) > self.allowed_loss*np.mean(self.termination.confidence.weights*self.termination.avg_loss):
            # too much loss added to termination classifier
            self.log("[option] Data loss: {} avg loss: {}".format(self.termination.confidence.weights*termination_loss, self.termination.confidence.weights*self.termination.avg_loss))
            self.log("[option] Termination loss too high. Data not added")
            return False

        self.log("[option] Samples from Markov classifier will be added")
        return True

    def train_initiation(
            self,
            embedding_epochs_per_cycle,
            classifier_epochs_per_cycle,
            num_cycles=1,
            shuffle_data=False):
        # train initiation classifier
        self.log("[option] Training initiation classifier...")
        print("[option] Training initiation classifier...")
        self.initiation.train(
            num_cycles,
            embedding_epochs_per_cycle,
            classifier_epochs_per_cycle,
            shuffle_data=shuffle_data
        )
        self.log("[option] Finished training initiation classifier")
        print("[option] Finished training initiation classifier")
        
    def train_termination(
            self,
            embedding_epochs_per_cycle,
            classifier_epochs_per_cycle,
            num_cycles=1,
            shuffle_data=False):
        # train termination classifier
        self.log("[option] Training termination classifier...")
        print("[option] Training termination classifier...")
        self.termination.train(
            num_cycles,
            embedding_epochs_per_cycle,
            classifier_epochs_per_cycle,
            shuffle_data=shuffle_data
        )
        self.log("[option] Finished training termination classifier")
        print("[option] Finished training termination classifier")

    def bootstrap_policy(
            self, 
            bootstrap_env,
            max_steps,
            success_rate_for_well_trained):
        # initial policy train
        # bootstrap_env: the environment the agent policy is trained on
        #       this environment should reset to an appropriate location and
        #       end episode (with appropriate reward) when the option policy has 
        #       hit a termination
        # max_steps: maximum number of steps the policy can take to train
        step_number = 0
        episode_number = 0
        total_reward = 0
        success_queue_size = 500
        success_rates = deque(maxlen=success_queue_size)
        self.log("[option] Bootstrapping option policy...")

        state, info = bootstrap_env.reset()

        while step_number < max_steps:
            # action selection
            action = self.policy.act(state)

            # step
            next_state, reward, done, info = bootstrap_env.step(action)
            #  really need to change this too
            self.policy.observe(state, action, reward, next_state, done)
            total_reward += reward
            step_number += 1
            state = next_state

            if done or info['needs_reset']:
                # check if env sent done signal and agent didn't die
                success_rates.append(done and not info['dead'])
                well_trained = len(success_rates) == success_queue_size \
                    and np.mean(success_rates) >= success_rate_for_well_trained

                if well_trained:
                    self.log('[option] Policy well trained in {} steps and {} episodes. Success rate {}'.format(step_number, episode_number, np.mean(success_rates)))
                    print('[option] Policy well trained in {} steps and {} episodes. Success rate {}'.format(step_number, episode_number, np.mean(success_rates)))
                    return

                episode_number += 1
                state, info = bootstrap_env.reset()
                if (episode_number - 1) % 50 == 0:
                    self.log("[option] Completed Episode {} steps {} success rate {}".format(episode_number-1, step_number, np.mean(success_rates)))
                    print("[option] Completed Episode {} steps {} success rate {}".format(episode_number-1, step_number, np.mean(success_rates)))

        self.log("[option] Policy did not reach well trained threshold. Success rate {}".format(np.mean(success_rates)))
        print("[option] Policy did not reach well trained threshold. Success rate {}".format(np.mean(success_rates)))

    def add_data_from_files_initiation(self, positive_files, negative_files, priority_negative_files):
        self.initiation.add_data_from_files(
            positive_files,
            negative_files,
            priority_negative_files
        )

    def add_data_from_files_termination(self, positive_files, negative_files, priority_negative_files):
        self.termination.add_data_from_files(
            positive_files,
            negative_files,
            priority_negative_files
        )

    def initiation_update_confidence(self, was_successful):
        self.initiation.update_confidence(was_successful)

    def termination_update_confidence(self, was_successful):
        self.termination.update_confidence(was_successful)

    def _option_success(
            self,
            positions,
            agent_space_states,
            markov_termination,
            agent_space_termination):
        
        # if we have a markoc classifier add data        
        if self.markov_classifier_idx is not None:
            self.markov_classifiers[self.markov_classifier_idx].add_positive(
                agent_space_states,
                positions
            )
            # test markov classifier to see if we can add data to global classifier
            if self._test_markov_classifier(
                agent_space_states,
                [agent_space_termination]
            ):
                self.initiation.add_data(
                    agent_space_states,
                    [agent_space_termination]
                )
                self.termination.add_data(
                    [agent_space_termination],
                    agent_space_states
                )
                self.initiation.update_confidence(was_successful=True)
                self.termination.update_confidence(was_successful=True)

        # if we don't have markov classifier add markov classifier
        else:
            self._add_markov_classifier(
                agent_space_states,
                positions,
                markov_termination,
            )

    def _option_fail(
            self,
            positions,
            agent_space_states):

        # if we have markov classifier add negative data to classifier
        if self.markov_classifier_idx is not None:
            self.markov_classifiers[self.markov_classifier_idx].add_negative(
                agent_space_states, positions
            )
        # if we don't have classifier add negative samples to global classifier (?)
        # maybe check if loss looks good (?)
        if self._test_markov_classifier(
            agent_space_states,
            [],
            test_negative_only=True
        ):
            self.initiation.add_data(
                negative_data=agent_space_states
            )
            self.initiation.update_confidence(was_successful=False)
        