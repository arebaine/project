import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.linalg import expm
from market_env import MarketSimulator
from train import DDPG_Trader

def evaluate_agent(agent, env, num_episodes=500, steps=2000):
    total_rewards = []
    
    for _ in range(num_episodes):
        # Initial trajectory purely to seed the GRU history window
        hist_signals = np.zeros((1, agent.W + 1))
        hist_signals[0, 0] = 1.0 # Starting value
        I_t = torch.zeros((1, 1))
        
        idx_theta = np.random.choice(3)
        theta = env.theta_vals[idx_theta]
        
        # Burn-in history
        for t in range(1, agent.W + 1):
            idx_theta = env._step_mc(idx_theta, env.P_theta)
            theta = env.theta_vals[idx_theta]
            dW = np.random.normal(0, np.sqrt(env.dt))
            dS = 3.0 * (theta - hist_signals[0, t-1]) * env.dt + 0.2 * dW
            hist_signals[0, t] = hist_signals[0, t-1] + dS

        episode_reward = 0
        
        for t in range(steps):
            history_tensor = torch.FloatTensor(hist_signals[:, 1:]).unsqueeze(-1)
            S_t = torch.FloatTensor(hist_signals[:, -1]).unsqueeze(-1)
            
            with torch.no_grad():
                state = agent._get_state_features(history_tensor, S_t, I_t)
                I_next = agent.actor(state)
            
            # Step environment forward
            idx_theta = env._step_mc(idx_theta, env.P_theta)
            theta = env.theta_vals[idx_theta]
            dW = np.random.normal(0, np.sqrt(env.dt))
            dS = 3.0 * (theta - hist_signals[0, -1]) * env.dt + 0.2 * dW
            S_next = hist_signals[0, -1] + dS
            
            # Calculate reward
            dq = I_next - I_t
            reward = I_next.item() * (S_next - S_t.item()) - agent.lambd * abs(dq.item())
            episode_reward += reward
            
            # Shift history
            hist_signals = np.roll(hist_signals, -1, axis=1)
            hist_signals[0, -1] = S_next
            I_t = I_next
            
        total_rewards.append(episode_reward)
        
    return np.mean(total_rewards), np.std(total_rewards)

if __name__ == "__main__":
    env = MarketSimulator(dt=0.2, W=10)
    
    methods = ['hid', 'prob', 'reg']
    
    for method in methods:
        print(f"\n--- Training {method}-DDPG ---")
        agent = DDPG_Trader(method=method)
        
        # Step 1: Pre-training for Two-Step methods
        if method in ['prob', 'reg']:
            print("Pre-training GRU Filter...")
            agent.pretrain_gru(env, iterations=5000)
            
        # Step 2: Main DDPG Training loop
        print("Training RL Agent...")
        N_train = 10000
        epsilon_start, epsilon_min = 100.0, 0.01
        
        for m in range(1, N_train + 1):
            eps = max(epsilon_start / (epsilon_start + m), epsilon_min)
            signals, inventories, _ = env.generate_batch(512, complexity=1)
            
            # Critic inner loop
            l_updates = 5 if method != 'hid' else 1
            for _ in range(l_updates):
                agent.train_ddpg_step(signals, inventories, eps)
                
            if m % 2000 == 0:
                print(f"Iteration {m}/{N_train} completed.")
                
        # Step 3: Evaluation
        print("Evaluating Agent...")
        mean_rev, std_rev = evaluate_agent(agent, env)
        print(f"{method}-DDPG Mean Reward: {mean_rev:.2f} +/- {std_rev:.2f}")