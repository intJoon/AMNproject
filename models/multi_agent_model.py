import pickle
import os
import numpy as np
from collections import deque
import random
import math

class MultiAgentDDPGLoadBalancer:
    def __init__(self, state_dim=24, action_dim=4, learning_rate=0.002, num_agents=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.num_agents = num_agents
        
        self.agent_q_tables = [{} for _ in range(num_agents)]
        self.cooperation_matrix = np.ones((num_agents, num_agents)) * 0.1
        
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.gamma = 0.95
    
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
    
    def choose_action(self, state, agent_id, training=True):
        state = np.array(state).flatten()
        
        if training and np.random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_key = self.get_state_key(state)
        
        if state_key not in self.agent_q_tables[agent_id]:
            self.agent_q_tables[agent_id][state_key] = np.zeros(self.action_dim)
        
        return np.argmax(self.agent_q_tables[agent_id][state_key])
    
    def get_cooperative_action(self, state, training=True):
        state = np.array(state).flatten()
        state_key = self.get_state_key(state)
        
        agent_actions = []
        agent_q_values = []
        
        for agent_id in range(self.num_agents):
            if state_key not in self.agent_q_tables[agent_id]:
                self.agent_q_tables[agent_id][state_key] = np.zeros(self.action_dim)
            
            action = self.choose_action(state, agent_id, training)
            q_value = self.agent_q_tables[agent_id][state_key][action]
            
            agent_actions.append(action)
            agent_q_values.append(q_value)
        
        cooperation_scores = np.zeros(self.action_dim)
        for action in range(self.action_dim):
            for agent_id in range(self.num_agents):
                if agent_actions[agent_id] == action:
                    cooperation_scores[action] += agent_q_values[agent_id]
                    for other_id in range(self.num_agents):
                        if other_id != agent_id:
                            cooperation_scores[action] += self.cooperation_matrix[agent_id][other_id] * agent_q_values[other_id]
        
        return np.argmax(cooperation_scores)
    
    def update_q_value(self, state, action, reward, next_state, done, agent_id):
        state = np.array(state).flatten()
        next_state = np.array(next_state).flatten()
        
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if state_key not in self.agent_q_tables[agent_id]:
            self.agent_q_tables[agent_id][state_key] = np.zeros(self.action_dim)
        
        if next_state_key not in self.agent_q_tables[agent_id]:
            self.agent_q_tables[agent_id][next_state_key] = np.zeros(self.action_dim)
        
        current_q = self.agent_q_tables[agent_id][state_key][action]
        next_max_q = np.max(self.agent_q_tables[agent_id][next_state_key])
        
        target_q = reward + (self.gamma * next_max_q * (1 - done))
        self.agent_q_tables[agent_id][state_key][action] = current_q + self.learning_rate * (target_q - current_q)
    
    def update_cooperation_matrix(self, agent_id, other_id, success):
        if success:
            self.cooperation_matrix[agent_id][other_id] = min(1.0, self.cooperation_matrix[agent_id][other_id] + 0.02)
        else:
            self.cooperation_matrix[agent_id][other_id] = max(0.0, self.cooperation_matrix[agent_id][other_id] - 0.005)
    
    def predict_server(self, features, server_mapping=None):
        if server_mapping is None:
            server_mapping = {0: 'h1', 1: 'h2', 2: 'h3', 3: 'h4'}
        
        state = np.array(features).flatten()
        action = self.get_cooperative_action(state, training=False)
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
                action = self.get_cooperative_action(state, training=True)
                
                true_server = server_mapping[y_train[i]]
                pred_server = servers[action]
                
                reward = 1.0 if pred_server == true_server else -1.0
                
                if i < len(X_train) - 1:
                    next_state = np.array(X_train[i + 1]).flatten()
                    done = False
                else:
                    next_state = state
                    done = True
                
                for agent_id in range(self.num_agents):
                    self.update_q_value(state, action, reward, next_state, done, agent_id)
                    
                    for other_id in range(self.num_agents):
                        if other_id != agent_id:
                            success = (pred_server == true_server)
                            self.update_cooperation_matrix(agent_id, other_id, success)
                
                total_reward += reward
                
                if pred_server == true_server:
                    correct += 1
            
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
                action = self.get_cooperative_action(state, training=False)
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
                'agent_q_tables': self.agent_q_tables,
                'cooperation_matrix': self.cooperation_matrix,
                'epsilon': self.epsilon,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'num_agents': self.num_agents
            }, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            state_dim=data['state_dim'],
            action_dim=data['action_dim'],
            num_agents=data.get('num_agents', 4)
        )
        instance.agent_q_tables = data['agent_q_tables']
        instance.cooperation_matrix = data['cooperation_matrix']
        instance.epsilon = data.get('epsilon', 0.01)
        return instance

