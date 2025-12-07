import pickle
import os
import numpy as np
from collections import defaultdict
import math

class TemporalGraphLoadBalancer:
    def __init__(self, hidden_dim=16, temporal_window=5):
        self.hidden_dim = hidden_dim
        self.temporal_window = temporal_window
        
        self.server_embeddings = None
        self.temporal_features = []
        self.graph_adjacency = None
        self.feature_names = None
        
    def build_temporal_graph(self, X, y, server_mapping):
        servers = ['h1', 'h2', 'h3', 'h4']
        n_servers = len(servers)
        
        adjacency = np.zeros((n_servers, n_servers))
        server_sequences = {server: [] for server in servers}
        
        for i in range(len(X)):
            true_server = server_mapping[y[i]]
            server_sequences[true_server].append(X[i])
        
        for i, server1 in enumerate(servers):
            for j, server2 in enumerate(servers):
                if i == j:
                    adjacency[i, j] = 1.0
                else:
                    if len(server_sequences[server1]) > 0 and len(server_sequences[server2]) > 0:
                        seq1 = np.array(server_sequences[server1][-self.temporal_window:])
                        seq2 = np.array(server_sequences[server2][-self.temporal_window:])
                        
                        if len(seq1) > 0 and len(seq2) > 0:
                            mean1 = np.mean(seq1, axis=0)
                            mean2 = np.mean(seq2, axis=0)
                            similarity = 1.0 / (1.0 + np.linalg.norm(mean1 - mean2))
                            adjacency[i, j] = similarity
        
        self.graph_adjacency = adjacency
        return adjacency
    
    def temporal_convolution(self, sequence):
        if len(sequence) == 0:
            return np.zeros(self.hidden_dim)
        
        sequence = np.array(sequence)
        if len(sequence.shape) == 1:
            sequence = sequence.reshape(1, -1)
        
        weights = np.exp(-np.arange(len(sequence)) * 0.1)
        weights = weights / weights.sum()
        
        weighted_features = np.dot(weights, sequence)
        
        if len(weighted_features) < self.hidden_dim:
            padded = np.pad(weighted_features, (0, self.hidden_dim - len(weighted_features)))
        else:
            padded = weighted_features[:self.hidden_dim]
        
        return padded
    
    def spatial_temporal_convolution(self, node_features, adjacency, temporal_sequence):
        temporal_features = self.temporal_convolution(temporal_sequence)
        
        feature_dim = node_features.shape[1]
        
        if self.server_embeddings is None:
            self.server_embeddings = np.random.randn(4, feature_dim) * 0.1
        elif self.server_embeddings.shape[1] != feature_dim:
            self.server_embeddings = np.random.randn(4, feature_dim) * 0.1
        
        enhanced_features = node_features + self.server_embeddings
        
        spatial_features = np.dot(adjacency, enhanced_features)
        
        temporal_dim = min(len(temporal_features), spatial_features.shape[1])
        combined = spatial_features.copy()
        combined[:, :temporal_dim] += temporal_features[:temporal_dim]
        
        return combined
    
    def predict_server(self, features, current_loads=None, server_mapping=None, temporal_history=None):
        if self.graph_adjacency is None:
            servers = ['h1', 'h2', 'h3', 'h4']
            loads = [current_loads.get(s, 0) if current_loads else 0 for s in servers]
            if all(l == 0 for l in loads):
                return 'h1'
            best_idx = np.argmin(loads)
            return servers[best_idx]
        
        if temporal_history is None:
            temporal_history = []
        
        servers = ['h1', 'h2', 'h3', 'h4']
        node_features = np.array([features] * 4)
        
        temporal_seq = temporal_history[-self.temporal_window:] if len(temporal_history) > 0 else [features]
        
        output = self.spatial_temporal_convolution(node_features, self.graph_adjacency, temporal_seq)
        
        server_scores = []
        for i in range(4):
            score = np.sum(output[i])
            if current_loads:
                score += current_loads.get(servers[i], 0) * 0.1
            server_scores.append(score)
        
        best_idx = np.argmin(server_scores)
        return servers[best_idx]
    
    def train(self, X_train, y_train, X_val=None, y_val=None, server_mapping=None, feature_names=None, epochs=50):
        self.feature_names = feature_names if feature_names is not None else []
        
        if server_mapping is None:
            server_mapping = {0: 'h1', 1: 'h2', 2: 'h3', 3: 'h4'}
        
        adjacency = self.build_temporal_graph(X_train, y_train, server_mapping)
        
        servers = ['h1', 'h2', 'h3', 'h4']
        learning_rate = 0.01
        temporal_history = []
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            
            for i in range(len(X_train)):
                features = X_train[i]
                true_server = server_mapping[y_train[i]]
                true_idx = servers.index(true_server)
                
                temporal_history.append(features)
                if len(temporal_history) > self.temporal_window * 10:
                    temporal_history.pop(0)
                
                temporal_seq = temporal_history[-self.temporal_window:] if len(temporal_history) >= self.temporal_window else temporal_history
                
                node_features = np.array([features] * 4)
                output = self.spatial_temporal_convolution(node_features, adjacency, temporal_seq)
                
                server_scores = [np.sum(output[j]) for j in range(4)]
                pred_idx = np.argmin(server_scores)
                
                loss = (server_scores[pred_idx] - server_scores[true_idx]) ** 2
                total_loss += loss
                
                if pred_idx == true_idx:
                    correct += 1
                
                if self.server_embeddings is not None:
                    feature_dim = self.server_embeddings.shape[1]
                    output_dim = output.shape[1]
                    
                    if output_dim == feature_dim:
                        gradient = (output[pred_idx] - output[true_idx]) * learning_rate
                    elif output_dim > feature_dim:
                        gradient = (output[pred_idx][:feature_dim] - output[true_idx][:feature_dim]) * learning_rate
                    else:
                        gradient = np.zeros(feature_dim, dtype=np.float64)
                        gradient[:output_dim] = (output[pred_idx] - output[true_idx]) * learning_rate
                    
                    gradient = np.asarray(gradient, dtype=np.float64).flatten()
                    self.server_embeddings[true_idx] = np.asarray(self.server_embeddings[true_idx], dtype=np.float64) - gradient * 0.1
                    self.server_embeddings[pred_idx] = np.asarray(self.server_embeddings[pred_idx], dtype=np.float64) + gradient * 0.1
            
            accuracy = correct / len(X_train) if len(X_train) > 0 else 0
            avg_loss = total_loss / len(X_train) if len(X_train) > 0 else 0
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
        
        if X_val is not None and y_val is not None:
            correct = 0
            temporal_history = []
            for i in range(len(X_val)):
                features = X_val[i]
                true_server = server_mapping[y_val[i]]
                
                temporal_history.append(features)
                if len(temporal_history) > self.temporal_window * 10:
                    temporal_history.pop(0)
                
                temporal_seq = temporal_history[-self.temporal_window:] if len(temporal_history) >= self.temporal_window else temporal_history
                
                pred_server = self.predict_server(features, None, server_mapping, temporal_seq)
                
                if pred_server == true_server:
                    correct += 1
            
            val_accuracy = correct / len(X_val) if len(X_val) > 0 else 0
            print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        return self
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'server_embeddings': self.server_embeddings,
                'graph_adjacency': self.graph_adjacency,
                'temporal_features': self.temporal_features,
                'hidden_dim': self.hidden_dim,
                'temporal_window': self.temporal_window,
                'feature_names': self.feature_names
            }, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            hidden_dim=data.get('hidden_dim', 16),
            temporal_window=data.get('temporal_window', 5)
        )
        instance.server_embeddings = data['server_embeddings']
        instance.graph_adjacency = data['graph_adjacency']
        instance.temporal_features = data.get('temporal_features', [])
        instance.feature_names = data.get('feature_names', [])
        return instance

