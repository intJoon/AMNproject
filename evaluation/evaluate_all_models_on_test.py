import sys
import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
from data_processing.preprocess_data import create_features, create_labels, prepare_model_features

def load_test_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file = os.path.join(base_dir, 'dataset', '[for testing]training_data_300samples_20251207_121637.json')
    
    print(f"Loading test data from: {test_file}")
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples")
    
    print("Preprocessing data...")
    df = create_features(df)
    df = create_labels(df)
    
    X, y, feature_names = prepare_model_features(df, exclude_server_column=True)
    
    server_mapping = {i: server for i, server in enumerate(['h1', 'h2', 'h3', 'h4'])}
    
    print(f"Features shape: {X.shape} (server column removed)")
    print(f"Labels shape: {y.shape}")
    print(f"Feature names: {len(feature_names)} features")
    
    return X, y, feature_names, server_mapping, df

def load_model(model_name, feature_names, server_mapping):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if model_name == 'round_robin':
        return RoundRobinLoadBalancer()
    elif model_name == 'random_forest':
        model = RandomForestLoadBalancer.load(
            os.path.join(base_dir, 'saved_models', 'random_forest.pkl')
        )
        model.server_mapping = server_mapping
        model.feature_names = feature_names
        return model
    elif model_name == 'graph_coloring':
        model = GraphColoringLoadBalancer.load(
            os.path.join(base_dir, 'saved_models', 'graph_coloring.pkl')
        )
        model.feature_names = feature_names
        return model
    elif model_name == 'ddpg':
        model = SimpleDDPGLoadBalancer.load(
            os.path.join(base_dir, 'saved_models', 'ddpg.pkl')
        )
        return model
    elif model_name == 'ppo':
        model = PPOLoadBalancer.load(
            os.path.join(base_dir, 'saved_models', 'ppo.pkl')
        )
        return model
    elif model_name == 'improved_ddpg':
        model = ImprovedDDPGLoadBalancer.load(
            os.path.join(base_dir, 'saved_models', 'improved_ddpg.pkl')
        )
        return model
    elif model_name == 'multi_agent':
        model = MultiAgentDDPGLoadBalancer.load(
            os.path.join(base_dir, 'saved_models', 'multi_agent.pkl')
        )
        return model
    elif model_name == 'transformer':
        model = TransformerLoadBalancer.load(
            os.path.join(base_dir, 'saved_models', 'transformer.pkl')
        )
        model.server_mapping = server_mapping
        model.feature_names = feature_names
        return model
    elif model_name == 'gat':
        model = GraphAttentionLoadBalancer.load(
            os.path.join(base_dir, 'saved_models', 'gat.pkl')
        )
        model.feature_names = feature_names
        return model
    elif model_name == 'tgnn':
        model = TemporalGraphLoadBalancer.load(
            os.path.join(base_dir, 'saved_models', 'tgnn.pkl')
        )
        model.feature_names = feature_names
        return model
    else:
        return None

def evaluate_model_on_test(model, X, y, server_mapping, model_name, feature_names):
    print(f"  Evaluating {model_name}...")
    
    try:
        if model_name == 'round_robin':
            predictions = model.predict(X)
            y_pred_encoded = np.array([list(server_mapping.values()).index(p) if p in server_mapping.values() else 0 
                                       for p in predictions])
        elif model_name == 'random_forest':
            y_pred_encoded = model.predict(X)
            predictions = [server_mapping.get(p, f'h{p+1}') for p in y_pred_encoded]
        elif model_name == 'graph_coloring':
            predictions = []
            current_loads = {}
            for features in X:
                pred_server = model.predict_server(features, current_loads, server_mapping)
                predictions.append(pred_server)
                current_loads[pred_server] = current_loads.get(pred_server, 0) + 1
            
            y_pred_encoded = np.array([list(server_mapping.values()).index(p) if p in server_mapping.values() else 0 
                                       for p in predictions])
        elif model_name == 'ddpg':
            predictions = []
            for features in X:
                pred_server = model.predict_server(features, server_mapping)
                predictions.append(pred_server)
            
            y_pred_encoded = np.array([list(server_mapping.values()).index(p) if p in server_mapping.values() else 0 
                                       for p in predictions])
        elif model_name in ['ppo', 'improved_ddpg', 'multi_agent']:
            predictions = []
            for features in X:
                pred_server = model.predict_server(features, server_mapping)
                predictions.append(pred_server)
            
            y_pred_encoded = np.array([list(server_mapping.values()).index(p) if p in server_mapping.values() else 0 
                                       for p in predictions])
        elif model_name == 'transformer':
            predictions = []
            history = []
            for features in X:
                pred_server, history = model.predict_server(features, server_mapping, history)
                predictions.append(pred_server)
            
            y_pred_encoded = np.array([list(server_mapping.values()).index(p) if p in server_mapping.values() else 0 
                                       for p in predictions])
        elif model_name in ['gat', 'tgnn']:
            predictions = []
            current_loads = {}
            temporal_history = []
            for features in X:
                if model_name == 'tgnn':
                    pred_server = model.predict_server(features, current_loads, server_mapping, temporal_history)
                    temporal_history.append(features)
                    if len(temporal_history) > 10:
                        temporal_history.pop(0)
                else:
                    pred_server = model.predict_server(features, current_loads, server_mapping)
                predictions.append(pred_server)
                current_loads[pred_server] = current_loads.get(pred_server, 0) + 1
            
            y_pred_encoded = np.array([list(server_mapping.values()).index(p) if p in server_mapping.values() else 0 
                                       for p in predictions])
        else:
            return None
        
        accuracy = accuracy_score(y, y_pred_encoded)
        
        true_servers = [server_mapping.get(label, f'h{label+1}') for label in y]
        
        response_times = {'h1': [], 'h2': [], 'h3': [], 'h4': []}
        if feature_names and 'response_time_ms' in feature_names:
            feature_idx = feature_names.index('response_time_ms')
            for i, server in enumerate(predictions):
                if server in response_times:
                    rt = X[i, feature_idx]
                    response_times[server].append(rt)
        
        avg_response_times = {server: np.mean(rt_list) if rt_list else 0 
                             for server, rt_list in response_times.items()}
        
        overall_avg_rt = np.mean([rt for rt_list in response_times.values() for rt in rt_list]) if any(response_times.values()) else 0
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'y_pred_encoded': y_pred_encoded,
            'true_servers': true_servers,
            'avg_response_times': avg_response_times,
            'overall_avg_rt': overall_avg_rt
        }
    except Exception as e:
        print(f"    Error evaluating {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_reports(results, df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)
    
    summary_data = []
    all_comparisons = []
    
    for model_name, result in results.items():
        if result is None:
            continue
        
        accuracy = result['accuracy']
        predictions = result['predictions']
        true_servers = result['true_servers']
        
        summary_data.append({
            'Model': model_name.replace('_', ' ').title(),
            'Accuracy': f"{accuracy:.4f}",
            'Accuracy (%)': f"{accuracy * 100:.2f}%",
            'Overall Avg RT (ms)': f"{result['overall_avg_rt']:.2f}"
        })
        
        comparison = pd.DataFrame({
            'Model': model_name,
            'Actual_Optimal': true_servers,
            'Predicted': predictions,
            'Correct': [t == p for t, p in zip(true_servers, predictions)]
        })
        
        all_comparisons.append(comparison)
        
        csv_path = os.path.join(output_dir, f'{model_name}_test_results.csv')
        comparison.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, 'test_evaluation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")
    
    all_comparisons_df = pd.concat(all_comparisons, ignore_index=True)
    all_path = os.path.join(output_dir, 'all_models_test_results.csv')
    all_comparisons_df.to_csv(all_path, index=False)
    print(f"  Saved: {all_path}")
    
    return summary_df

def main():
    print("=" * 80)
    print("Test Dataset Validation: All 10 Models")
    print("=" * 80)
    
    X, y, feature_names, server_mapping, df = load_test_data()
    
    print(f"\nTest dataset size: {len(X)} samples")
    print(f"Number of features: {len(feature_names)}")
    print(f"Server mapping: {server_mapping}\n")
    
    models_to_evaluate = [
        'round_robin',
        'random_forest',
        'graph_coloring',
        'ddpg',
        'ppo',
        'improved_ddpg',
        'multi_agent',
        'transformer',
        'gat',
        'tgnn'
    ]
    
    results = {}
    
    print("=" * 80)
    print("EVALUATING MODELS")
    print("=" * 80)
    
    for model_name in models_to_evaluate:
        print(f"\nLoading {model_name}...")
        try:
            model = load_model(model_name, feature_names, server_mapping)
            if model:
                result = evaluate_model_on_test(model, X, y, server_mapping, model_name, feature_names)
                if result:
                    results[model_name] = result
                    print(f"  ✓ {model_name}: Accuracy = {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
                else:
                    print(f"  ✗ {model_name}: Evaluation failed")
                    results[model_name] = None
            else:
                print(f"  ✗ {model_name}: Model not found")
                results[model_name] = None
        except Exception as e:
            print(f"  ✗ {model_name}: Error - {str(e)}")
            results[model_name] = None
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    baseline_accuracy = results.get('round_robin', {}).get('accuracy', 0) if results.get('round_robin') else 0
    
    print(f"\n{'Model':<25} {'Accuracy':<15} {'Accuracy (%)':<15} {'Improvement':<15}")
    print("-" * 80)
    
    for model_name in models_to_evaluate:
        if model_name in results and results[model_name]:
            result = results[model_name]
            accuracy = result['accuracy']
            accuracy_pct = accuracy * 100
            
            if baseline_accuracy > 0 and model_name != 'round_robin':
                improvement = ((accuracy - baseline_accuracy) / baseline_accuracy * 100)
                improvement_str = f"{improvement:+.2f}%"
            else:
                improvement_str = "Baseline"
            
            display_name = model_name.replace('_', ' ').title()
            print(f"{display_name:<25} {accuracy:<15.4f} {accuracy_pct:<15.2f}% {improvement_str:<15}")
    
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    
    for model_name in models_to_evaluate:
        if model_name in results and results[model_name]:
            result = results[model_name]
            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  Accuracy: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
            print(f"  Overall Average Response Time: {result['overall_avg_rt']:.2f} ms")
            print(f"  Server-specific Average Response Times:")
            for server, avg_rt in result['avg_response_times'].items():
                print(f"    {server}: {avg_rt:.2f} ms")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'evaluation', 'test_results')
    
    summary_df = generate_reports(results, df, output_dir)
    
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print(f"Total models evaluated: {len([r for r in results.values() if r is not None])}")
    
    return results

if __name__ == '__main__':
    results = main()

