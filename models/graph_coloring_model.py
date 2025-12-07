import pickle
import os
import numpy as np
from collections import defaultdict
import math

class GraphColoringLoadBalancer:
    def __init__(self):
        self.server_weights = None
        self.server_history = defaultdict(list)
        self.conflict_graph = None
        self.server_mapping = None
        self.reverse_mapping = None
        
    def build_conflict_graph(self, X, y, server_mapping):
        servers = ['h1', 'h2', 'h3', 'h4']
        n_servers = len(servers)
        
        conflict_matrix = np.zeros((n_servers, n_servers))
        server_stats = {server: {'mean_rt': 0, 'count': 0, 'features': []} for server in servers}
        
        for i in range(len(X)):
            true_server = server_mapping[y[i]]
            
            server_stats[true_server]['count'] += 1
            server_stats[true_server]['features'].append(X[i])
            
            if self.feature_names and 'mean_rt' in self.feature_names:
                rt_idx = self.feature_names.index('mean_rt')
                server_stats[true_server]['mean_rt'] += X[i, rt_idx]
        
        for server in servers:
            if server_stats[server]['count'] > 0:
                server_stats[server]['mean_rt'] /= server_stats[server]['count']
                server_stats[server]['mean_features'] = np.mean(server_stats[server]['features'], axis=0)
            else:
                server_stats[server]['mean_features'] = np.zeros(X.shape[1])
        
        for i, server1 in enumerate(servers):
            for j, server2 in enumerate(servers):
                if i != j:
                    weight_diff = abs(server_stats[server1]['mean_rt'] - server_stats[server2]['mean_rt'])
                    feature_diff = np.linalg.norm(server_stats[server1]['mean_features'] - server_stats[server2]['mean_features'])
                    conflict_matrix[i, j] = weight_diff + feature_diff * 0.1
        
        self.conflict_graph = conflict_matrix
        self.server_weights = server_stats
        
    def color_servers(self, features, current_loads):
        servers = ['h1', 'h2', 'h3', 'h4']
        
        server_scores = {}
        max_load = max(current_loads.values()) if current_loads.values() else 1
        has_any_feature_info = False
        
        for i, server in enumerate(servers):
            if self.server_weights and server in self.server_weights and self.server_weights[server]['count'] > 0:
                has_any_feature_info = True
                break
        
        for i, server in enumerate(servers):
            load = current_loads.get(server, 0)
            
            feature_match_score = 0.0
            has_feature_info = False
            if self.server_weights and server in self.server_weights and self.server_weights[server]['count'] > 0:
                if 'mean_features' in self.server_weights[server]:
                    mean_features = self.server_weights[server]['mean_features']
                    feature_diff = features - mean_features
                    feature_distance = np.linalg.norm(feature_diff)
                    mean_norm = np.linalg.norm(mean_features)
                    if mean_norm > 1e-8:
                        feature_match_score = feature_distance / (mean_norm + 1e-8)
                    else:
                        feature_match_score = feature_distance
                    has_feature_info = True
                elif self.feature_names and 'mean_rt' in self.feature_names:
                    rt_idx = self.feature_names.index('mean_rt')
                    if rt_idx < len(features):
                        feature_rt = features[rt_idx]
                        weight = self.server_weights[server]['mean_rt']
                        if weight > 1e-8:
                            feature_match_score = abs(feature_rt - weight) / (weight + 1e-8)
                        else:
                            feature_match_score = abs(feature_rt - weight) + 1.0
                        has_feature_info = True
            
            conflict_penalty = 0
            if self.conflict_graph is not None:
                for j, other_server in enumerate(servers):
                    if i != j:
                        other_load = current_loads.get(other_server, 0)
                        conflict_penalty += self.conflict_graph[i, j] * other_load * 0.1
            
            load_score = load / (max_load + 1) if max_load > 0 else 0
            
            if has_any_feature_info and has_feature_info:
                score = load_score * 0.4 + feature_match_score * 0.5 + conflict_penalty * 0.1
            else:
                score = load_score * 0.9 + conflict_penalty * 0.1
            
            server_scores[server] = score
        
        if len(server_scores) == 0:
            return 'h1'
        
        best_server = min(server_scores, key=server_scores.get)
        return best_server
    
    def predict_server(self, features, current_loads=None, server_mapping=None):
        if current_loads is None:
            current_loads = {}
        
        if self.server_weights is None:
            servers = ['h1', 'h2', 'h3', 'h4']
            loads = [current_loads.get(s, 0) for s in servers]
            if all(l == 0 for l in loads):
                return 'h1'
            best_idx = np.argmin(loads)
            return servers[best_idx]
        
        result = self.color_servers(features, current_loads)
        if result is None or result not in ['h1', 'h2', 'h3', 'h4']:
            servers = ['h1', 'h2', 'h3', 'h4']
            loads = [current_loads.get(s, 0) for s in servers]
            best_idx = np.argmin(loads)
            return servers[best_idx]
        return result
    
    def predict(self, X):
        predictions = []
        current_loads = defaultdict(int)
        
        for features in X:
            server = self.predict_server(features, dict(current_loads))
            if self.reverse_mapping:
                label = self.reverse_mapping.get(server, 0)
                predictions.append(label)
            else:
                predictions.append(server)
            current_loads[server] += 1
        
        return np.array(predictions)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, server_mapping=None, feature_names=None):
        self.feature_names = feature_names if feature_names is not None else []
        
        if server_mapping is None:
            server_mapping = {0: 'h1', 1: 'h2', 2: 'h3', 3: 'h4'}
        
        self.server_mapping = server_mapping
        self.reverse_mapping = {v: k for k, v in server_mapping.items()}
        
        self.build_conflict_graph(X_train, y_train, server_mapping)
        
        if X_val is not None and y_val is not None:
            correct = 0
            total = len(X_val)
            
            for i in range(len(X_val)):
                true_server = server_mapping[y_val[i]]
                features = X_val[i]
                
                current_loads = defaultdict(int)
                for j in range(i):
                    prev_server = server_mapping.get(y_val[j], None)
                    if prev_server:
                        current_loads[prev_server] += 1
                
                server_loads = {}
                for server in ['h1', 'h2', 'h3', 'h4']:
                    server_loads[server] = current_loads.get(server, 0)
                
                pred_server = self.predict_server(features, server_loads, server_mapping)
                
                if pred_server is None or pred_server not in ['h1', 'h2', 'h3', 'h4']:
                    servers = ['h1', 'h2', 'h3', 'h4']
                    loads = [server_loads.get(s, 0) for s in servers]
                    if all(l == 0 for l in loads):
                        pred_server = 'h1'
                    else:
                        best_idx = np.argmin(loads)
                        pred_server = servers[best_idx]
                
                if pred_server == true_server:
                    correct += 1
            
            accuracy = correct / total if total > 0 else 0
            print(f"Validation Accuracy: {accuracy:.4f}")
        
        return self
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'server_weights': self.server_weights,
                'conflict_graph': self.conflict_graph,
                'feature_names': self.feature_names,
                'server_mapping': self.server_mapping,
                'reverse_mapping': self.reverse_mapping
            }, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.server_weights = data['server_weights']
        instance.conflict_graph = data['conflict_graph']
        instance.feature_names = data.get('feature_names', [])
        instance.server_mapping = data.get('server_mapping', {0: 'h1', 1: 'h2', 2: 'h3', 3: 'h4'})
        instance.reverse_mapping = data.get('reverse_mapping', {v: k for k, v in instance.server_mapping.items()})
        return instance

