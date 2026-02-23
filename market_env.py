import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.linalg import expm

class MarketSimulator:
    def __init__(self, dt=0.2, W=10, I_min=-10, I_max=10):
        self.dt = dt
        self.W = W
        self.I_min = I_min
        self.I_max = I_max
        
        # Transition matrices (A) and probabilities (P = exp(A * dt))
        self.A_theta = np.array([[-0.1, 0.05, 0.05], [0.05, -0.1, 0.05], [0.05, 0.05, -0.1]])
        self.P_theta = expm(self.A_theta * self.dt)
        self.theta_vals = np.array([0.9, 1.0, 1.1])
        
        self.A_kappa = np.array([[-0.1, 0.1], [0.1, -0.1]])
        self.P_kappa = expm(self.A_kappa * self.dt)
        self.kappa_vals = np.array([3.0, 7.0])
        
        self.A_sigma = np.array([[-0.1, 0.1], [0.1, -0.1]])
        self.P_sigma = expm(self.A_sigma * self.dt)
        self.sigma_vals = np.array([0.1, 0.3])

    def _step_mc(self, current_idx, P_matrix):
        return np.random.choice(len(P_matrix), p=P_matrix[current_idx])

    def generate_batch(self, batch_size, complexity=3):
        """
        Complexity 1: theta MC
        Complexity 2: theta, kappa MCs
        Complexity 3: theta, kappa, sigma MCs
        Returns signals (b, W+2), inventories (b, 1), and true theta regimes (b, W+2)
        """
        signals = np.zeros((batch_size, self.W + 2))
        regimes = np.zeros((batch_size, self.W + 2), dtype=int)
        
        # Random initial inventory
        inventories = np.random.uniform(self.I_min, self.I_max, size=(batch_size, 1))
        
        for b in range(batch_size):
            # Init states
            idx_theta = np.random.choice(3)
            idx_kappa = np.random.choice(2)
            idx_sigma = np.random.choice(2)
            
            theta = self.theta_vals[idx_theta]
            kappa = self.kappa_vals[idx_kappa] if complexity >= 2 else 3.0
            sigma = self.sigma_vals[idx_sigma] if complexity == 3 else 0.2
            
            # Invariant invariant mean/vol as starting point
            inv_vol = sigma / (2 * kappa)
            signals[b, 0] = np.random.normal(1.0, 3 * inv_vol)
            regimes[b, 0] = idx_theta
            
            for t in range(1, self.W + 2):
                idx_theta = self._step_mc(idx_theta, self.P_theta)
                theta = self.theta_vals[idx_theta]
                
                if complexity >= 2:
                    idx_kappa = self._step_mc(idx_kappa, self.P_kappa)
                    kappa = self.kappa_vals[idx_kappa]
                if complexity == 3:
                    idx_sigma = self._step_mc(idx_sigma, self.P_sigma)
                    sigma = self.sigma_vals[idx_sigma]
                
                dW = np.random.normal(0, np.sqrt(self.dt))
                # Euler-Maruyama
                dS = kappa * (theta - signals[b, t-1]) * self.dt + sigma * dW
                signals[b, t] = signals[b, t-1] + dS
                regimes[b, t] = idx_theta
                
        return torch.FloatTensor(signals), torch.FloatTensor(inventories), torch.LongTensor(regimes)