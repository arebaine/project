import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.linalg import expm

class GRUNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=20, num_layers=5, output_type='hidden'):
        super().__init__()
        self.output_type = output_type # 'hidden', 'prob', or 'pred'
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
        if self.output_type == 'prob':
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 3), # 3 regimes
                nn.Softmax(dim=-1)
            )
        elif self.output_type == 'pred' or self.output_type == 'hidden_pred':
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.SiLU(),
                nn.Linear(64, 1) # Next signal
            )

    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :] # Take last output
        
        if self.output_type == 'hidden':
            return last_hidden
        elif self.output_type == 'hidden_pred':
            pred = self.ffn(last_hidden)
            return last_hidden, pred
        else:
            return self.ffn(last_hidden)

class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim=64, I_max=10):
        super().__init__()
        self.I_max = I_max
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, state):
        return self.net(state) * self.I_max

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim), # +1 for action (inventory)
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)