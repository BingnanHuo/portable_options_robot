import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from portable.option.ensemble.criterion import batched_L_divergence
from portable.option.ensemble.attention import Attention 
from portable.option.policy.models import LinearQFunction, compute_q_learning_loss


class ValueEnsemble():

    def __init__(self, 
        device,
        embedding_output_size=64, 
        gru_hidden_size=128,
        learning_rate=2.5e-4,
        discount_rate=0.9,
        num_modules=8, 
        num_output_classes=18,
        plot_dir=None,
        verbose=False,):
        
        self.num_modules = num_modules
        self.num_output_classes = num_output_classes
        self.device = device
        self.gamma = discount_rate
        self.verbose = verbose

        self.embedding = Attention(
            embedding_size=embedding_output_size, 
            num_attention_modules=self.num_modules, 
            plot_dir=plot_dir
        ).to(self.device)

        self.recurrent_memory = nn.GRU(
            input_size=embedding_output_size,
            hidden_size=gru_hidden_size,
            batch_first=True,
        ).to(self.device)

        self.q_networks = nn.ModuleList(
            [LinearQFunction(in_features=gru_hidden_size, n_actions=num_output_classes) for _ in range(self.num_modules)]
        ).to(self.device)
        self.target_q_networks = deepcopy(self.q_networks)
        self.target_q_networks.eval()

        self.optimizer = optim.Adam(
            list(self.embedding.parameters()) + list(self.q_networks.parameters()) + list(self.recurrent_memory.parameters()),
            learning_rate,
        )

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.embedding.state_dict(), os.path.join(path, 'embedding.pt'))
        torch.save(self.q_networks.state_dict(), os.path.join(path, 'policy_networks.pt'))

    def load(self, path):
        self.embedding.load_state_dict(torch.load(os.path.join(path, 'embedding.pt')))
        self.q_networks.load_state_dict(torch.load(os.path.join(path, 'policy_networks.pt')))

    def train(self, exp_batch, errors_out=None, update_target_network=False, plot_embedding=False):
        """
        update both the embedding network and the value network by backproping
        the sumed divergence and q learning loss
        """
        self.embedding.train()
        self.q_networks.train()
        self.recurrent_memory.flatten_parameters()

        batch_states = exp_batch['state']
        batch_actions = exp_batch['action']
        batch_rewards = exp_batch['reward']
        batch_next_states = exp_batch['next_state']
        batch_dones = exp_batch['is_state_terminal']

        loss = 0

        # divergence loss
        state_embeddings = self.embedding(batch_states, return_attention_mask=False, plot=plot_embedding)  # (batch_size, num_modules, embedding_size)
        state_embeddings, _ = self.recurrent_memory(state_embeddings)  # (batch_size, num_modules, gru_out_size)
        l_div = batched_L_divergence(state_embeddings)
        loss += l_div

        # q learning loss
        td_losses = np.zeros((self.num_modules,))
        next_state_embeddings = self.embedding(batch_next_states, return_attention_mask=False)
        next_state_embeddings, _ = self.recurrent_memory(next_state_embeddings)

        # keep track of all error out for each module 
        all_errors_out = np.zeros((self.num_modules, len(batch_states)))

        for idx in range(self.num_modules):

            # predicted q values
            state_attention = state_embeddings[:,idx,:]  # (batch_size, emb_out_size)
            batch_pred_q_all_actions = self.q_networks[idx](state_attention)  # (batch_size, num_actions)
            batch_pred_q = batch_pred_q_all_actions.evaluate_actions(batch_actions)  # (batch_size,)

            # target q values 
            with torch.no_grad():
                next_state_attention = next_state_embeddings[:,idx,:]  # (batch_size, emb_out_size)
                batch_next_state_q_all_actions = self.target_q_networks[idx](next_state_attention)  # (batch_size, num_actions)
                next_state_values = batch_next_state_q_all_actions.max  # (batch_size,)
                batch_q_target = batch_rewards + self.gamma * (1-batch_dones) *  next_state_values # (batch_size,)
            
            # loss
            td_loss = compute_q_learning_loss(exp_batch, batch_pred_q, batch_q_target, errors_out=errors_out)
            all_errors_out[idx] = errors_out
            loss += td_loss
            if self.verbose: td_losses[idx] = td_loss.item()

        # update errors_out, so it accounts for all modules in ensemble
        del errors_out[:]
        avg_errors_out = np.mean(all_errors_out, axis=0)
        for e in avg_errors_out:
            errors_out.append(e)
    
        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        if update_target_network:
            self.target_q_networks.load_state_dict(self.q_networks.state_dict())

        # logging
        if self.verbose:
            # for idx in range(self.num_modules):
            #     print("\t - Value {}: loss {:.6f}".format(idx, td_losses[idx]))
            print(f"Div loss: {l_div.item()}. Q loss: {np.sum(td_losses)}")

        self.embedding.eval()
        self.q_networks.eval()

    def predict_actions(self, state, return_q_values=False):
        """
        given a state, each one in the ensemble predicts an action
        args:
            return_q_values: if True, return the predicted q values each learner predicts on the action of their choice.
        """
        self.embedding.eval()
        self.q_networks.eval()
        with torch.no_grad():
            embeddings = self.embedding(state, return_attention_mask=False).detach()
            self.recurrent_memory.flatten_parameters()
            embeddings, _ = self.recurrent_memory(embeddings)

            actions = np.zeros(self.num_modules, dtype=np.int)
            q_values = np.zeros(self.num_modules, dtype=np.float)
            for idx in range(self.num_modules):
                attention = embeddings[:,idx,:]
                q_vals = self.q_networks[idx](attention)
                actions[idx] = q_vals.greedy_actions
                q_values[idx] = q_vals.max

        if return_q_values:
            return actions, q_values
        return actions

    def get_attention(self, x):
        self.embedding.eval()
        x = x.to(self.device)
        _, atts = self.embedding(x, return_attention_mask=True).detach()
        return atts