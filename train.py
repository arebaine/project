import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.linalg import expm
from models import GRUNetwork, Actor, Critic

class DDPG_Trader:
    def __init__(self, method, W=10, lr=0.001, gamma=0.99, lambd=0.05, I_max=10):
        self.method = method
        self.W = W
        self.gamma = gamma
        self.lambd = lambd
        self.I_max = I_max
        
        # Determine state dimension based on method
        if method == 'hid':
            self.state_dim = 10 + 2 # gru_hidden (10) + S_t (1) + I_t (1)
            self.gru = GRUNetwork(hidden_dim=10, num_layers=1, output_type='hidden_pred')
        elif method == 'prob':
            self.state_dim = 3 + 2 # probs (3) + S_t (1) + I_t (1)
            self.gru = GRUNetwork(hidden_dim=20, num_layers=5, output_type='prob')
        elif method == 'reg':
            self.state_dim = 1 + 2 # next_S (1) + S_t (1) + I_t (1)
            self.gru = GRUNetwork(hidden_dim=20, num_layers=5, output_type='pred')

        self.actor = Actor(self.state_dim, I_max=self.I_max)
        self.critic = Critic(self.state_dim)
        self.critic_tgt = Critic(self.state_dim)
        self.critic_tgt.load_state_dict(self.critic.state_dict())
        
        self.gru_opt = optim.Adam(self.gru.parameters(), lr=lr)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

    def _get_state_features(self, history, S_t, I_t):
        if self.method == 'hid':
            hidden, _ = self.gru(history)
            features = hidden
        else:
            with torch.no_grad():
                features = self.gru(history)
        return torch.cat([features, S_t, I_t], dim=1)

    def pretrain_gru(self, env, iterations=5000, batch_size=512):
        """Used for two-step prob-DDPG and reg-DDPG."""
        if self.method == 'hid': return
        criterion = nn.CrossEntropyLoss() if self.method == 'prob' else nn.MSELoss()
        
        for i in range(iterations):
            signals, _, regimes = env.generate_batch(batch_size)
            # Input is W sequence ending at t
            history = signals[:, :-2].unsqueeze(-1) 
            target_regime = regimes[:, -2]
            target_S = signals[:, -1].unsqueeze(-1)

            out = self.gru(history)
            loss = criterion(out, target_regime) if self.method == 'prob' else criterion(out, target_S)
            
            self.gru_opt.zero_grad()
            loss.backward()
            self.gru_opt.step()

    def train_ddpg_step(self, signals, inventories, epsilon):
        batch_size = signals.shape[0]
        
        # S_t is at index W, S_{t+1} is at W+1
        history_t = signals[:, :-2].unsqueeze(-1)
        S_t = signals[:, -2].unsqueeze(-1)
        S_next = signals[:, -1].unsqueeze(-1)
        I_t = inventories
        
        # hid-DDPG trains the GRU to predict S_{t+1} first in the loop
        if self.method == 'hid':
            _, S_pred = self.gru(history_t)
            gru_loss = nn.MSELoss()(S_pred, S_next)
            self.gru_opt.zero_grad()
            gru_loss.backward()
            self.gru_opt.step()

        # Build State G_t
        state = self._get_state_features(history_t, S_t, I_t).detach()
        
        # Execute Action with noise
        with torch.no_grad():
            action_clean = self.actor(state)
            noise = torch.randn_like(action_clean) * epsilon
            I_next = torch.clamp(action_clean + noise, -self.I_max, self.I_max)

        # Compute Reward
        dq = I_next - I_t
        reward = I_next * (S_next - S_t) - self.lambd * torch.abs(dq)

        # Build Next State G_{t+1}
        history_next = signals[:, 1:-1].unsqueeze(-1)
        state_next = self._get_state_features(history_next, S_next, I_next).detach()

        # Update Critic
        with torch.no_grad():
            next_action = self.actor(state_next)
            target_q = reward + self.gamma * self.critic_tgt(state_next, next_action)
            
        current_q = self.critic(state, I_next)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Update Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft update target
        for param, target_param in zip(self.critic.parameters(), self.critic_tgt.parameters()):
            target_param.data.copy_(0.001 * param.data + (1 - 0.001) * target_param.data)