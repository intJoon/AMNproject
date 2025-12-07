import pickle
import os
import numpy as np
from collections import deque
import random
import math

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0
    
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(left + 1, s - self.tree[left])
    
    def total(self):
        return self.tree[0]
    
    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class ImprovedDDPGLoadBalancer:
    def __init__(self, state_dim=24, action_dim=4, learning_rate=0.001, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.q_table = {}
        self.memory = SumTree(10000)
        self.max_priority = 1.0
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
    
    def get_state_key(self, state):
        try:
            state_arr = np.asarray(state, dtype=float)
            if state_arr.ndim == 0:
                state_arr = np.array([state_arr])
            state_arr = state_arr.flatten()
            state_rounded = np.round(state_arr, decimals=1)
            return tuple(state_rounded)
        except Exception as e:
            state_arr = np.array([float(x) for x in np.array(state).flatten()])
            state_rounded = np.round(state_arr, decimals=1)
            return tuple(state_rounded)
    
    def choose_action(self, state, training=True):
        state = np.array(state).flatten()
        
        if training and np.random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_key = self.get_state_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        
        return np.argmax(self.q_table[state_key])
    
    def store_transition(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        transition = (state_key, action, reward, next_state, done)
        priority = self.max_priority
        self.memory.add(priority, transition)
    
    def sample_batch(self, batch_size=32):
        batch = []
        idxs = []
        segment = self.memory.total() / batch_size
        priorities = []
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.memory.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        
        sampling_probabilities = priorities / self.memory.total()
        is_weight = np.power(self.memory.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        
        return batch, idxs, is_weight
    
    def update_q_value(self, state, action, reward, next_state, done, td_error):
        state = np.array(state).flatten()
        next_state = np.array(next_state).flatten()
        
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_dim)
        
        current_q = self.q_table[state_key][action]
        next_max_q = np.max(self.q_table[next_state_key])
        
        target_q = reward + (self.gamma * next_max_q * (1 - done))
        self.q_table[state_key][action] = current_q + self.learning_rate * (target_q - current_q)
        
        priority = (abs(td_error) + 1e-6) ** self.alpha
        self.max_priority = max(self.max_priority, priority)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
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
            total_reward = 0
            correct = 0
            
            for i in range(len(X_train)):
                state = np.array(X_train[i]).flatten()
                action = self.choose_action(state, training=True)
                
                true_server = server_mapping[y_train[i]]
                pred_server = servers[action]
                
                reward = 1.0 if pred_server == true_server else -1.0
                
                if i < len(X_train) - 1:
                    next_state = np.array(X_train[i + 1]).flatten()
                    done = False
                else:
                    next_state = state
                    done = True
                
                state_key = self.get_state_key(state)
                next_state_key = self.get_state_key(next_state)
                
                if state_key not in self.q_table:
                    self.q_table[state_key] = np.zeros(self.action_dim)
                if next_state_key not in self.q_table:
                    self.q_table[next_state_key] = np.zeros(self.action_dim)
                
                current_q = self.q_table[state_key][action]
                next_max_q = np.max(self.q_table[next_state_key])
                target_q = reward + (self.gamma * next_max_q * (1 - done))
                td_error = target_q - current_q
                
                self.store_transition(state, action, reward, next_state, done)
                self.update_q_value(state, action, reward, next_state, done, td_error)
                
                total_reward += reward
                
                if pred_server == true_server:
                    correct += 1
            
            accuracy = correct / len(X_train) if len(X_train) > 0 else 0
            avg_reward = total_reward / len(X_train) if len(X_train) > 0 else 0
            
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
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'alpha': self.alpha,
                'beta': self.beta
            }, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            state_dim=data['state_dim'],
            action_dim=data['action_dim'],
            alpha=data.get('alpha', 0.6),
            beta=data.get('beta', 0.4)
        )
        instance.q_table = data['q_table']
        instance.epsilon = data.get('epsilon', 0.01)
        return instance

