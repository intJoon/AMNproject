import pickle
import os
import numpy as np
from collections import deque
import random
import math

class PPOLoadBalancer:
    def __init__(self, state_dim=24, action_dim=4, learning_rate=0.001, clip_epsilon=0.2, gamma=0.95, gae_lambda=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.policy_table = {}
        self.value_table = {}
        self.memory = []
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
    
    def get_state_key(self, state):
        try:
            state_arr = np.asarray(state, dtype=float)
            if state_arr.ndim == 0:
                state_arr = np.array([state_arr])
            state_arr = state_arr.flatten()
            state_rounded = np.round(state_arr, decimals=2)
            return tuple(state_rounded)
        except Exception as e:
            state_arr = np.array([float(x) for x in np.array(state).flatten()])
            state_rounded = np.round(state_arr, decimals=2)
            return tuple(state_rounded)
    
    def get_policy(self, state):
        state_key = self.get_state_key(state)
        if state_key not in self.policy_table:
            self.policy_table[state_key] = np.ones(self.action_dim) / self.action_dim
        policy = self.policy_table[state_key].copy()
        policy = np.clip(policy, 1e-8, 1.0)
        policy_sum = policy.sum()
        if policy_sum > 0:
            policy = policy / policy_sum
        else:
            policy = np.ones(self.action_dim) / self.action_dim
        self.policy_table[state_key] = policy
        return policy
    
    def get_value(self, state):
        state_key = self.get_state_key(state)
        if state_key not in self.value_table:
            self.value_table[state_key] = 0.0
        return self.value_table[state_key]
    
    def choose_action(self, state, training=True):
        state = np.array(state).flatten()
        
        if training and np.random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        policy = self.get_policy(state)
        policy = np.asarray(policy, dtype=np.float64)
        policy = np.clip(policy, 1e-8, 1.0)
        policy_sum = policy.sum()
        if abs(policy_sum - 1.0) > 1e-6:
            policy = policy / policy_sum
        
        action = np.random.choice(self.action_dim, p=policy)
        return action
    
    def compute_gae(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_values[step] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def update_policy(self, states, actions, old_probs, advantages, returns, epochs=10):
        for _ in range(epochs):
            for state, action, old_prob, adv, ret in zip(states, actions, old_probs, advantages, returns):
                state_key = self.get_state_key(state)
                policy = self.get_policy(state)
                
                prob = policy[action]
                ratio = prob / (old_prob + 1e-8)
                
                clipped_ratio = np.clip(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                policy_loss = -min(ratio * adv, clipped_ratio * adv)
                
                normalized_adv = np.clip(adv, -5.0, 5.0)
                policy_loss = policy_loss * normalized_adv
                
                new_policy = policy.copy()
                update = self.learning_rate * policy_loss
                update = np.clip(update, -0.5, 0.5)
                new_policy[action] = np.clip(new_policy[action] + update, 1e-8, 1.0)
                new_policy = np.clip(new_policy, 1e-8, 1.0)
                policy_sum = new_policy.sum()
                if policy_sum > 0:
                    new_policy = new_policy / policy_sum
                else:
                    new_policy = np.ones(self.action_dim) / self.action_dim
                self.policy_table[state_key] = new_policy
                
                value = self.get_value(state)
                value_update = self.learning_rate * 1.0 * (ret - value)
                value_update = np.clip(value_update, -1.0, 1.0)
                self.value_table[state_key] = value + value_update
    
    def predict_server(self, features, server_mapping=None):
        if server_mapping is None:
            server_mapping = {0: 'h1', 1: 'h2', 2: 'h3', 3: 'h4'}
        
        state = np.array(features).flatten()
        action = self.choose_action(state, training=False)
        return server_mapping.get(action, f'h{action + 1}')
    
    def train(self, X_train, y_train, X_val=None, y_val=None, server_mapping=None, epochs=50):
        if server_mapping is None:
            server_mapping = {0: 'h1', 1: 'h2', 2: 'h3', 3: 'h4'}
        
        servers = ['h1', 'h2', 'h3', 'h4']
        
        for epoch in range(epochs):
            current_lr = self.initial_learning_rate * (0.995 ** epoch)
            self.learning_rate = max(current_lr, self.initial_learning_rate * 0.2)
            
            total_reward = 0
            correct = 0
            
            states = []
            actions = []
            rewards = []
            old_probs = []
            values = []
            next_values = []
            dones = []
            
            for i in range(len(X_train)):
                state = np.array(X_train[i]).flatten()
                policy = self.get_policy(state)
                action = self.choose_action(state, training=True)
                old_prob = policy[action]
                
                true_server = server_mapping[y_train[i]]
                pred_server = servers[action]
                
                if pred_server == true_server:
                    reward = 5.0
                else:
                    reward = -1.0
                
                value = self.get_value(state)
                
                if i < len(X_train) - 1:
                    next_state = np.array(X_train[i + 1]).flatten()
                    next_value = self.get_value(next_state)
                    done = False
                else:
                    next_state = state
                    next_value = 0.0
                    done = True
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                old_probs.append(old_prob)
                values.append(value)
                next_values.append(next_value)
                dones.append(done)
                
                total_reward += reward
                
                if pred_server == true_server:
                    correct += 1
            
            if len(states) > 0:
                advantages, returns = self.compute_gae(rewards, values, next_values, dones)
                advantages = np.array(advantages)
                if advantages.std() > 1e-8:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                advantages = np.clip(advantages, -5.0, 5.0)
                self.update_policy(states, actions, old_probs, advantages.tolist(), returns, epochs=6)
            
            accuracy = correct / len(X_train) if len(X_train) > 0 else 0
            avg_reward = total_reward / len(X_train) if len(X_train) > 0 else 0
            
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {accuracy:.4f}, Avg Reward: {avg_reward:.4f}, Epsilon: {self.epsilon:.4f}")
        
        if X_val is not None and y_val is not None:
            correct = 0
            for i in range(len(X_val)):
                state = np.array(X_val[i]).flatten()
                action = self.choose_action(state, training=False)
                true_server = server_mapping[y_val[i]]
                pred_server = servers[action]
                
                if pred_server == true_server:
                    correct += 1
            
            val_accuracy = correct / len(X_val) if len(X_val) > 0 else 0
            print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        return self
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'policy_table': self.policy_table,
                'value_table': self.value_table,
                'epsilon': self.epsilon,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim
            }, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            state_dim=data['state_dim'],
            action_dim=data['action_dim']
        )
        instance.policy_table = data['policy_table']
        instance.value_table = data['value_table']
        instance.epsilon = data.get('epsilon', 0.01)
        return instance

