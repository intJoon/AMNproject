import pickle
import os
import numpy as np
from collections import defaultdict
import math

class GraphAttentionLoadBalancer:
    def __init__(self, num_heads=4, hidden_dim=16, dropout=0.1):
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        self.server_embeddings = None
        self.attention_weights = None
        self.graph_adjacency = None
        self.feature_names = None
        self.server_projection = None
        
    def build_graph_adjacency(self, X, y, server_mapping):
        servers = ['h1', 'h2', 'h3', 'h4']
        n_servers = len(servers)
        
        adjacency = np.zeros((n_servers, n_servers))
        server_features = {server: [] for server in servers}
        
        for i in range(len(X)):
            true_server = server_mapping[y[i]]
            server_features[true_server].append(X[i])
        
        for i, server1 in enumerate(servers):
            for j, server2 in enumerate(servers):
                if i == j:
                    adjacency[i, j] = 1.0
                else:
                    if len(server_features[server1]) > 0 and len(server_features[server2]) > 0:
                        feat1 = np.mean(server_features[server1], axis=0)
                        feat2 = np.mean(server_features[server2], axis=0)
                        similarity = 1.0 / (1.0 + np.linalg.norm(feat1 - feat2))
                        adjacency[i, j] = similarity
        
        self.graph_adjacency = adjacency
        return adjacency
    
    def multi_head_attention(self, node_features, adjacency):
        n_nodes = node_features.shape[0]
        feature_dim = node_features.shape[1]
        temperature = np.sqrt(feature_dim)
        head_outputs = []
        
        for head in range(self.num_heads):
            attention_scores = np.zeros((n_nodes, n_nodes))
            
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if adjacency[i, j] > 0:
                        score = np.dot(node_features[i], node_features[j]) / temperature
                        attention_scores[i, j] = score
            
            attention_scores = attention_scores * adjacency
            attention_scores = np.where(adjacency > 0, attention_scores, -np.inf)
            
            max_scores = np.max(attention_scores, axis=1, keepdims=True)
            exp_scores = np.exp(np.clip(attention_scores - max_scores, -500, 500))
            attention_weights = exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-8)
            attention_weights = attention_weights * adjacency
            
            head_output = np.dot(attention_weights, node_features)
            head_outputs.append(head_output)
        
        multi_head_output = np.concatenate(head_outputs, axis=1)
        return multi_head_output, attention_weights
    
    def forward(self, node_features, adjacency):
        feature_dim = node_features.shape[1]
        
        if self.server_embeddings is None or self.server_embeddings.shape[1] != feature_dim:
            self.server_embeddings = np.random.randn(4, feature_dim) * 0.1
        
        if self.server_embeddings.shape[1] != feature_dim:
            self.server_embeddings = np.random.randn(4, feature_dim) * 0.1
        
        enhanced_features = node_features + self.server_embeddings
        output, attention = self.multi_head_attention(enhanced_features, adjacency)
        
        return output, attention
    
    def predict_server(self, features, current_loads=None, server_mapping=None):
        if self.graph_adjacency is None:
            servers = ['h1', 'h2', 'h3', 'h4']
            loads = [current_loads.get(s, 0) if current_loads else 0 for s in servers]
            if all(l == 0 for l in loads):
                return 'h1'
            best_idx = np.argmin(loads)
            return servers[best_idx]
        
        servers = ['h1', 'h2', 'h3', 'h4']
        node_features = np.array([features] * 4)
        
        output, attention = self.forward(node_features, self.graph_adjacency)
        
        if self.server_projection is None:
            output_dim = output.shape[1]
            self.server_projection = np.random.randn(output_dim, 4) * 0.1
        
        server_logits = np.dot(output, self.server_projection)
        server_scores = np.array([server_logits[i, i] for i in range(4)])
        server_scores = np.asarray(server_scores, dtype=np.float64)
        
        if current_loads:
            for i, server in enumerate(servers):
                server_scores[i] += current_loads.get(server, 0) * 0.1
        
        best_idx = np.argmax(server_scores)
        return servers[best_idx]
    
    def train(self, X_train, y_train, X_val=None, y_val=None, server_mapping=None, feature_names=None, epochs=50):
        self.feature_names = feature_names if feature_names is not None else []
        
        if server_mapping is None:
            server_mapping = {0: 'h1', 1: 'h2', 2: 'h3', 3: 'h4'}
        
        adjacency = self.build_graph_adjacency(X_train, y_train, server_mapping)
        
        servers = ['h1', 'h2', 'h3', 'h4']
        initial_learning_rate = 0.01
        feature_dim = X_train.shape[1] if len(X_train) > 0 else 24
        
        if self.server_embeddings is None or self.server_embeddings.shape[1] != feature_dim:
            self.server_embeddings = np.random.randn(4, feature_dim) * 0.1
        
        for epoch in range(epochs):
            learning_rate = initial_learning_rate * (0.95 ** (epoch // 5))
            learning_rate = max(learning_rate, initial_learning_rate * 0.1)
            
            total_loss = 0
            correct = 0
            
            projection_gradients_accum = None
            embedding_gradients_accum = None
            batch_size = 0
            
            indices = np.random.permutation(len(X_train))
            
            for idx in indices:
                i = idx
                features = X_train[i]
                true_server = server_mapping[y_train[i]]
                true_idx = servers.index(true_server)
                
                node_features = np.array([features] * 4)
                output, attention = self.forward(node_features, adjacency)
                
                if self.server_projection is None:
                    output_dim = output.shape[1]
                    self.server_projection = np.random.randn(output_dim, 4) * 0.1
                
                server_logits = np.dot(output, self.server_projection)
                server_scores = np.array([server_logits[i, i] for i in range(4)])
                server_scores = np.asarray(server_scores, dtype=np.float64)
                pred_idx = np.argmax(server_scores)
                
                target_scores = np.zeros(4, dtype=np.float64)
                target_scores[true_idx] = 1.0
                max_score = float(np.max(server_scores))
                server_scores_shifted = np.asarray(server_scores, dtype=np.float64) - max_score
                server_scores_shifted = np.clip(server_scores_shifted, -500, 500)
                pred_scores = np.exp(server_scores_shifted)
                pred_scores = pred_scores / (np.sum(pred_scores) + 1e-8)
                
                loss = -np.sum(target_scores * np.log(pred_scores + 1e-8))
                total_loss += loss
                
                if pred_idx == true_idx:
                    correct += 1
                
                loss_gradient = pred_scores - target_scores
                
                projection_gradient = np.zeros_like(self.server_projection)
                for j in range(4):
                    projection_gradient[:, j] = output[j] * loss_gradient[j]
                
                embedding_gradient = np.zeros_like(self.server_embeddings)
                for j in range(4):
                    if j == true_idx:
                        embedding_gradient[j] = -features * (1.0 - pred_scores[j])
                    else:
                        embedding_gradient[j] = features * pred_scores[j]
                
                if projection_gradients_accum is None:
                    projection_gradients_accum = projection_gradient.copy()
                    embedding_gradients_accum = embedding_gradient.copy()
                else:
                    projection_gradients_accum += projection_gradient
                    embedding_gradients_accum += embedding_gradient
                
                batch_size += 1
                
                if batch_size >= 16 or i == indices[-1]:
                    projection_gradients_accum = projection_gradients_accum / batch_size
                    embedding_gradients_accum = embedding_gradients_accum / batch_size
                    
                    projection_gradients_accum = np.clip(projection_gradients_accum * learning_rate, -1.0, 1.0)
                    self.server_projection = np.asarray(self.server_projection, dtype=np.float64) - projection_gradients_accum
                    self.server_projection = np.clip(self.server_projection, -5.0, 5.0)
                    
                    embedding_gradients_accum = np.clip(embedding_gradients_accum * learning_rate * 0.5, -0.5, 0.5)
                    self.server_embeddings = np.asarray(self.server_embeddings, dtype=np.float64) + embedding_gradients_accum
                    self.server_embeddings = np.clip(self.server_embeddings, -5.0, 5.0)
                    
                    projection_gradients_accum = None
                    embedding_gradients_accum = None
                    batch_size = 0
            
            accuracy = correct / len(X_train) if len(X_train) > 0 else 0
            avg_loss = total_loss / len(X_train) if len(X_train) > 0 else 0
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
        
        if X_val is not None and y_val is not None:
            correct = 0
            for i in range(len(X_val)):
                features = X_val[i]
                true_server = server_mapping[y_val[i]]
                pred_server = self.predict_server(features, None, server_mapping)
                
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
                'attention_weights': self.attention_weights,
                'server_projection': self.server_projection,
                'num_heads': self.num_heads,
                'hidden_dim': self.hidden_dim,
                'feature_names': self.feature_names
            }, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            num_heads=data.get('num_heads', 4),
            hidden_dim=data.get('hidden_dim', 16)
        )
        instance.server_embeddings = data['server_embeddings']
        instance.graph_adjacency = data['graph_adjacency']
        instance.attention_weights = data.get('attention_weights')
        instance.server_projection = data.get('server_projection')
        instance.feature_names = data.get('feature_names', [])
        return instance

