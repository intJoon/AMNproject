import sys
import os
import pickle
import numpy as np
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.random_forest_model import RandomForestLoadBalancer
from models.transformer_model import TransformerLoadBalancer
from models.graph_coloring_model import GraphColoringLoadBalancer
from models.gat_model import GraphAttentionLoadBalancer
from models.tgnn_model import TemporalGraphLoadBalancer
from models.ddpg_model import SimpleDDPGLoadBalancer
from models.ppo_model import PPOLoadBalancer
from models.improved_ddpg_model import ImprovedDDPGLoadBalancer
from models.multi_agent_model import MultiAgentDDPGLoadBalancer
from evaluation.round_robin import RoundRobinLoadBalancer

class MininetLoadBalancer:
    def __init__(self, algorithm='random_forest'):
        self.algorithm = algorithm
        self.model = None
        self.server_mapping = {0: 'h1', 1: 'h2', 2: 'h3', 3: 'h4'}
        self.feature_names = None
        self.server_stats = defaultdict(lambda: {'count': 0, 'total_rt': 0, 'avg_rt': 0})
        self.current_loads = defaultdict(int)
        self.temporal_history = []
        self.transformer_history = []
        
        self.load_model()
    
    def load_model(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if self.algorithm == 'round_robin':
            self.model = RoundRobinLoadBalancer()
        elif self.algorithm == 'random_forest':
            model_path = os.path.join(base_dir, 'saved_models', 'random_forest.pkl')
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            self.model = data['model']
            
            pkl_path = os.path.join(base_dir, 'saved_models', 'preprocessed_data.pkl')
            with open(pkl_path, 'rb') as f:
                preprocessed = pickle.load(f)
            
            self.feature_names = preprocessed['feature_names']
            self.server_mapping = preprocessed['server_mapping']
        elif self.algorithm == 'graph_coloring':
            model_path = os.path.join(base_dir, 'saved_models', 'graph_coloring.pkl')
            self.model = GraphColoringLoadBalancer.load(model_path)
            
            pkl_path = os.path.join(base_dir, 'saved_models', 'preprocessed_data.pkl')
            with open(pkl_path, 'rb') as f:
                preprocessed = pickle.load(f)
            
            self.feature_names = preprocessed['feature_names']
            self.server_mapping = preprocessed['server_mapping']
        elif self.algorithm == 'ddpg':
            model_path = os.path.join(base_dir, 'saved_models', 'ddpg.pkl')
            self.model = SimpleDDPGLoadBalancer.load(model_path)
            
            pkl_path = os.path.join(base_dir, 'saved_models', 'preprocessed_data.pkl')
            with open(pkl_path, 'rb') as f:
                preprocessed = pickle.load(f)
            
            self.server_mapping = preprocessed['server_mapping']
        elif self.algorithm == 'ppo':
            model_path = os.path.join(base_dir, 'saved_models', 'ppo.pkl')
            self.model = PPOLoadBalancer.load(model_path)
            
            pkl_path = os.path.join(base_dir, 'saved_models', 'preprocessed_data.pkl')
            with open(pkl_path, 'rb') as f:
                preprocessed = pickle.load(f)
            
            self.server_mapping = preprocessed['server_mapping']
        elif self.algorithm == 'improved_ddpg':
            model_path = os.path.join(base_dir, 'saved_models', 'improved_ddpg.pkl')
            self.model = ImprovedDDPGLoadBalancer.load(model_path)
            
            pkl_path = os.path.join(base_dir, 'saved_models', 'preprocessed_data.pkl')
            with open(pkl_path, 'rb') as f:
                preprocessed = pickle.load(f)
            
            self.server_mapping = preprocessed['server_mapping']
        elif self.algorithm == 'multi_agent':
            model_path = os.path.join(base_dir, 'saved_models', 'multi_agent.pkl')
            self.model = MultiAgentDDPGLoadBalancer.load(model_path)
            
            pkl_path = os.path.join(base_dir, 'saved_models', 'preprocessed_data.pkl')
            with open(pkl_path, 'rb') as f:
                preprocessed = pickle.load(f)
            
            self.server_mapping = preprocessed['server_mapping']
        elif self.algorithm == 'transformer':
            model_path = os.path.join(base_dir, 'saved_models', 'transformer.pkl')
            self.model = TransformerLoadBalancer.load(model_path)
            
            pkl_path = os.path.join(base_dir, 'saved_models', 'preprocessed_data.pkl')
            with open(pkl_path, 'rb') as f:
                preprocessed = pickle.load(f)
            
            self.feature_names = preprocessed['feature_names']
            self.server_mapping = preprocessed['server_mapping']
        elif self.algorithm == 'gat':
            model_path = os.path.join(base_dir, 'saved_models', 'gat.pkl')
            self.model = GraphAttentionLoadBalancer.load(model_path)
            
            pkl_path = os.path.join(base_dir, 'saved_models', 'preprocessed_data.pkl')
            with open(pkl_path, 'rb') as f:
                preprocessed = pickle.load(f)
            
            self.feature_names = preprocessed['feature_names']
            self.server_mapping = preprocessed['server_mapping']
        elif self.algorithm == 'tgnn':
            model_path = os.path.join(base_dir, 'saved_models', 'tgnn.pkl')
            self.model = TemporalGraphLoadBalancer.load(model_path)
            
            pkl_path = os.path.join(base_dir, 'saved_models', 'preprocessed_data.pkl')
            with open(pkl_path, 'rb') as f:
                preprocessed = pickle.load(f)
            
            self.feature_names = preprocessed['feature_names']
            self.server_mapping = preprocessed['server_mapping']
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def extract_features(self, server_stats_dict):
        if self.feature_names is None:
            return None
        
        features = []
        for feature_name in self.feature_names:
            if feature_name == 'server_encoded':
                features.append(0)
            elif feature_name == 'response_time_ms':
                avg_rt = np.mean([stats['avg_rt'] for stats in server_stats_dict.values() if stats['count'] > 0])
                features.append(avg_rt if avg_rt > 0 else 145.0)
            elif feature_name == 'success':
                features.append(1.0)
            elif feature_name.startswith('h') and '_recent' in feature_name:
                server = feature_name.split('_')[0]
                if 'mean' in feature_name:
                    features.append(server_stats_dict.get(server, {}).get('avg_rt', 0))
                elif 'count' in feature_name:
                    features.append(server_stats_dict.get(server, {}).get('count', 0))
                elif 'success_rate' in feature_name:
                    features.append(1.0)
            elif feature_name in ['mean_rt', 'std_rt', 'min_rt', 'max_rt', 'count', 'success_rate']:
                if feature_name == 'mean_rt':
                    avg_rt = np.mean([stats['avg_rt'] for stats in server_stats_dict.values() if stats['count'] > 0])
                    features.append(avg_rt if avg_rt > 0 else 145.0)
                elif feature_name == 'count':
                    total_count = sum([stats['count'] for stats in server_stats_dict.values()])
                    features.append(total_count)
                else:
                    features.append(0.0)
            elif feature_name in ['hour', 'minute', 'second']:
                from datetime import datetime
                now = datetime.now()
                if feature_name == 'hour':
                    features.append(float(now.hour))
                elif feature_name == 'minute':
                    features.append(float(now.minute))
                elif feature_name == 'second':
                    features.append(float(now.second))
            else:
                features.append(0.0)
        
        return np.array(features)
    
    def select_server(self, current_server_stats=None):
        if current_server_stats is None:
            current_server_stats = self.server_stats
        
        if self.algorithm == 'round_robin':
            return self.model.get_next_server()
        
        features = self.extract_features(current_server_stats)
        
        if features is None:
            servers = ['h1', 'h2', 'h3', 'h4']
            loads = [self.current_loads.get(s, 0) for s in servers]
            best_idx = np.argmin(loads)
            return servers[best_idx]
        
        if self.algorithm == 'random_forest':
            features_2d = features.reshape(1, -1)
            pred_class = self.model.predict(features_2d)[0]
            return self.server_mapping.get(pred_class, f'h{pred_class + 1}')
        elif self.algorithm == 'graph_coloring':
            return self.model.predict_server(features, dict(self.current_loads), self.server_mapping)
        elif self.algorithm == 'ddpg':
            return self.model.predict_server(features, self.server_mapping)
        elif self.algorithm in ['ppo', 'improved_ddpg', 'multi_agent']:
            return self.model.predict_server(features, self.server_mapping)
        elif self.algorithm == 'transformer':
            pred_server, self.transformer_history = self.model.predict_server(features, self.server_mapping, self.transformer_history)
            return pred_server
        elif self.algorithm == 'gat':
            return self.model.predict_server(features, dict(self.current_loads), self.server_mapping)
        elif self.algorithm == 'tgnn':
            pred_server = self.model.predict_server(features, dict(self.current_loads), self.server_mapping, self.temporal_history)
            self.temporal_history.append(features)
            if len(self.temporal_history) > 10:
                self.temporal_history.pop(0)
            return pred_server
        else:
            return 'h1'
    
    def update_server_stats(self, server, response_time_ms, success=True):
        self.server_stats[server]['count'] += 1
        self.server_stats[server]['total_rt'] += response_time_ms
        self.server_stats[server]['avg_rt'] = (
            self.server_stats[server]['total_rt'] / self.server_stats[server]['count']
        )
        self.current_loads[server] += 1
    
    def get_server_loads(self):
        return dict(self.current_loads)
    
    def reset_loads(self):
        self.current_loads = defaultdict(int)
        self.temporal_history = []
        self.transformer_history = []
        if hasattr(self.model, 'reset'):
            self.model.reset()

def get_load_balancer(algorithm='random_forest'):
    return MininetLoadBalancer(algorithm=algorithm)

if __name__ == '__main__':
    print("Mininet Load Balancer - Test Mode")
    print("=" * 60)
    
    algorithms = ['round_robin', 'random_forest', 'graph_coloring', 'ddpg', 'ppo', 'improved_ddpg', 'multi_agent', 'transformer', 'gat', 'tgnn']
    
    for algo in algorithms:
        print(f"\nTesting {algo}...")
        lb = MininetLoadBalancer(algorithm=algo)
        
        for i in range(10):
            server = lb.select_server()
            response_time = np.random.normal(150, 20) if server in ['h1', 'h3'] else np.random.normal(600, 50)
            lb.update_server_stats(server, response_time)
            print(f"  Request {i+1}: {server} (response time: {response_time:.2f}ms)")
        
        print(f"  Server loads: {lb.get_server_loads()}")
        print(f"  Server stats:")
        for server, stats in lb.server_stats.items():
            if stats['count'] > 0:
                print(f"    {server}: avg_rt={stats['avg_rt']:.2f}ms, count={stats['count']}")

