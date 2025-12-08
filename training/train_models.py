import sys
import os
import pickle
import numpy as np

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

def load_preprocessed_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkl_path = os.path.join(base_dir, 'saved_models', 'preprocessed_data.pkl')
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def train_random_forest(data):
    print("=" * 60)
    print("Training Random Forest Model")
    print("=" * 60)
    
    model = RandomForestLoadBalancer(
        n_estimators=100, 
        max_depth=5,
        min_samples_split=15,
        min_samples_leaf=8,
        max_features='sqrt',
        class_weight='balanced'
    )
    model.server_mapping = data['server_mapping']
    model.feature_names = data['feature_names']
    
    model.train(
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val']
    )
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'saved_models', 'random_forest.pkl')
    model.save(model_path)
    print(f"Model saved to: {model_path}\n")
    
    return model

def train_graph_coloring(data):
    print("=" * 60)
    print("Training Graph Coloring Model")
    print("=" * 60)
    
    model = GraphColoringLoadBalancer()
    
    model.train(
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val'],
        data['server_mapping'],
        data['feature_names']
    )
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'saved_models', 'graph_coloring.pkl')
    model.save(model_path)
    print(f"Model saved to: {model_path}\n")
    
    return model

def train_ddpg(data):
    print("=" * 60)
    print("Training DDPG Model")
    print("=" * 60)
    
    model = SimpleDDPGLoadBalancer(
        state_dim=data['X_train'].shape[1],
        action_dim=4
    )
    
    model.train(
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val'],
        data['server_mapping'],
        epochs=50
    )
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'saved_models', 'ddpg.pkl')
    model.save(model_path)
    print(f"Model saved to: {model_path}\n")
    
    return model

def train_ppo(data):
    print("=" * 60)
    print("Training PPO Model")
    print("=" * 60)
    
    model = PPOLoadBalancer(
        state_dim=data['X_train'].shape[1],
        action_dim=4,
        learning_rate=0.001
    )
    
    model.train(
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val'],
        data['server_mapping'],
        epochs=50
    )
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'saved_models', 'ppo.pkl')
    model.save(model_path)
    print(f"Model saved to: {model_path}\n")
    
    return model

def train_improved_ddpg(data):
    print("=" * 60)
    print("Training Improved DDPG Model (SumTree)")
    print("=" * 60)
    
    model = ImprovedDDPGLoadBalancer(
        state_dim=data['X_train'].shape[1],
        action_dim=4
    )
    
    model.train(
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val'],
        data['server_mapping'],
        epochs=50
    )
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'saved_models', 'improved_ddpg.pkl')
    model.save(model_path)
    print(f"Model saved to: {model_path}\n")
    
    return model

def train_multi_agent(data):
    print("=" * 60)
    print("Training Multi-Agent DDPG Model")
    print("=" * 60)
    
    model = MultiAgentDDPGLoadBalancer(
        state_dim=data['X_train'].shape[1],
        action_dim=4,
        num_agents=4
    )
    
    model.train(
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val'],
        data['server_mapping'],
        epochs=50
    )
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'saved_models', 'multi_agent.pkl')
    model.save(model_path)
    print(f"Model saved to: {model_path}\n")
    
    return model

def train_transformer(data):
    print("=" * 60)
    print("Training Transformer Model")
    print("=" * 60)
    
    model = TransformerLoadBalancer(
        d_model=24,
        n_heads=4,
        n_layers=2,
        seq_length=5
    )
    
    model.train(
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val'],
        data['server_mapping'],
        data['feature_names'],
        epochs=30
    )
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'saved_models', 'transformer.pkl')
    model.save(model_path)
    print(f"Model saved to: {model_path}\n")
    
    return model

def train_gat(data):
    print("=" * 60)
    print("Training Graph Attention Network (GAT) Model")
    print("=" * 60)
    
    feature_dim = data['X_train'].shape[1]
    model = GraphAttentionLoadBalancer(
        num_heads=2,
        hidden_dim=min(16, feature_dim)
    )
    
    model.train(
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val'],
        data['server_mapping'],
        data['feature_names'],
        epochs=20
    )
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'saved_models', 'gat.pkl')
    model.save(model_path)
    print(f"Model saved to: {model_path}\n")
    
    return model

def train_tgnn(data):
    print("=" * 60)
    print("Training Temporal Graph Neural Network (TGNN) Model")
    print("=" * 60)
    
    model = TemporalGraphLoadBalancer(
        hidden_dim=8,
        temporal_window=3
    )
    
    model.train(
        data['X_train'],
        data['y_train'],
        data['X_val'],
        data['y_val'],
        data['server_mapping'],
        data['feature_names'],
        epochs=10
    )
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'saved_models', 'tgnn.pkl')
    model.save(model_path)
    print(f"Model saved to: {model_path}\n")
    
    return model

def main():
    print("Loading preprocessed data...")
    data = load_preprocessed_data()
    
    print(f"Training set: {data['X_train'].shape[0]} samples")
    print(f"Validation set: {data['X_val'].shape[0]} samples")
    print(f"Test set: {data['X_test'].shape[0]} samples\n")
    
    models = {}
    
    print("\nTraining baseline models...")
    models['random_forest'] = train_random_forest(data)
    models['graph_coloring'] = train_graph_coloring(data)
    models['ddpg'] = train_ddpg(data)
    
    print("\nTraining advanced reinforcement learning models...")
    models['ppo'] = train_ppo(data)
    models['improved_ddpg'] = train_improved_ddpg(data)
    models['multi_agent'] = train_multi_agent(data)
    
    print("\nTraining transformer and graph-based models...")
    models['transformer'] = train_transformer(data)
    models['gat'] = train_gat(data)
    models['tgnn'] = train_tgnn(data)
    
    print("=" * 60)
    print("All models trained successfully!")
    print(f"Total models: {len(models)}")
    print("=" * 60)

if __name__ == '__main__':
    main()

