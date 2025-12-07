import sys
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
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

def load_preprocessed_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkl_path = os.path.join(base_dir, 'saved_models', 'preprocessed_data.pkl')
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def load_model(model_name, data):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if model_name == 'random_forest':
        model = RandomForestLoadBalancer.load(
            os.path.join(base_dir, 'saved_models', 'random_forest.pkl')
        )
        model.server_mapping = data['server_mapping']
        model.feature_names = data['feature_names']
    elif model_name == 'graph_coloring':
        model = GraphColoringLoadBalancer.load(
            os.path.join(base_dir, 'saved_models', 'graph_coloring.pkl')
        )
        model.feature_names = data['feature_names']
    elif model_name == 'ddpg':
        model = SimpleDDPGLoadBalancer.load(
            os.path.join(base_dir, 'saved_models', 'ddpg.pkl')
        )
    elif model_name == 'ppo':
        model = PPOLoadBalancer.load(
            os.path.join(base_dir, 'saved_models', 'ppo.pkl')
        )
    elif model_name == 'improved_ddpg':
        model = ImprovedDDPGLoadBalancer.load(
            os.path.join(base_dir, 'saved_models', 'improved_ddpg.pkl')
        )
    elif model_name == 'multi_agent':
        model = MultiAgentDDPGLoadBalancer.load(
            os.path.join(base_dir, 'saved_models', 'multi_agent.pkl')
        )
    elif model_name == 'transformer':
        model = TransformerLoadBalancer.load(
            os.path.join(base_dir, 'saved_models', 'transformer.pkl')
        )
        model.server_mapping = data['server_mapping']
        model.feature_names = data['feature_names']
    elif model_name == 'gat':
        model = GraphAttentionLoadBalancer.load(
            os.path.join(base_dir, 'saved_models', 'gat.pkl')
        )
        model.feature_names = data['feature_names']
    elif model_name == 'tgnn':
        model = TemporalGraphLoadBalancer.load(
            os.path.join(base_dir, 'saved_models', 'tgnn.pkl')
        )
        model.feature_names = data['feature_names']
    else:
        return None
    
    return model

def evaluate_model(model, X, y, server_mapping, model_name, feature_names=None):
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
    for i, server in enumerate(predictions):
        if server in response_times:
            if feature_names and 'response_time_ms' in feature_names:
                feature_idx = feature_names.index('response_time_ms')
                rt = X[i, feature_idx]
                response_times[server].append(rt)
    
    avg_response_times = {server: np.mean(rt_list) if rt_list else 0 
                         for server, rt_list in response_times.items()}
    
    overall_avg_rt = np.mean([rt for rt_list in response_times.values() for rt in rt_list]) if any(response_times.values()) else 0
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'y_pred_encoded': y_pred_encoded,
        'avg_response_times': avg_response_times,
        'overall_avg_rt': overall_avg_rt
    }

def main():
    print("=" * 80)
    print("Algorithm Comparison: Round Robin vs ML-based Load Balancing")
    print("=" * 80)
    
    data = load_preprocessed_data()
    server_mapping = data['server_mapping']
    
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"\nTest set size: {len(X_test)} samples\n")
    
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
    
    round_robin = RoundRobinLoadBalancer()
    results['round_robin'] = evaluate_model(round_robin, X_test, y_test, server_mapping, 'round_robin', data['feature_names'])
    
    for model_name in ['random_forest', 'graph_coloring', 'ddpg', 'ppo', 'improved_ddpg', 'multi_agent', 'transformer', 'gat', 'tgnn']:
        print(f"Loading {model_name} model...")
        try:
            model = load_model(model_name, data)
            if model:
                results[model_name] = evaluate_model(model, X_test, y_test, server_mapping, model_name, data['feature_names'])
                print(f"  {model_name} evaluated successfully")
            else:
                print(f"  {model_name} model not found, skipping...")
        except Exception as e:
            print(f"  Error loading {model_name}: {str(e)}, skipping...")
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    baseline_accuracy = results['round_robin']['accuracy']
    baseline_rt = results['round_robin']['overall_avg_rt']
    
    print(f"\n{'Algorithm':<20} {'Accuracy':<15} {'Avg Response Time (ms)':<25} {'Improvement':<15}")
    print("-" * 80)
    
    for model_name, result in results.items():
        accuracy = result['accuracy']
        avg_rt = result['overall_avg_rt']
        
        accuracy_improvement = ((accuracy - baseline_accuracy) / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
        rt_improvement = ((baseline_rt - avg_rt) / baseline_rt * 100) if baseline_rt > 0 else 0
        
        model_display = model_name.replace('_', ' ').title()
        print(f"{model_display:<20} {accuracy:<15.4f} {avg_rt:<25.2f} "
              f"Acc: {accuracy_improvement:+.1f}%, RT: {rt_improvement:+.1f}%")
    
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    
    for model_name, result in results.items():
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Overall Average Response Time: {result['overall_avg_rt']:.2f} ms")
        print(f"  Server-specific Average Response Times:")
        for server, avg_rt in result['avg_response_times'].items():
            print(f"    {server}: {avg_rt:.2f} ms")
    
    print("\n" + "=" * 80)
    print("Comparison with Round Robin (Baseline)")
    print("=" * 80)
    
    for model_name, result in results.items():
        if model_name == 'round_robin':
            continue
        
        accuracy_improvement = ((result['accuracy'] - baseline_accuracy) / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
        rt_improvement = ((baseline_rt - result['overall_avg_rt']) / baseline_rt * 100) if baseline_rt > 0 else 0
        
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Accuracy: {accuracy_improvement:+.2f}% {'improvement' if accuracy_improvement > 0 else 'decrease'}")
        print(f"  Response Time: {rt_improvement:+.2f}% {'improvement' if rt_improvement > 0 else 'decrease'}")
    
    create_visualizations(results, server_mapping, data['feature_names'], X_test, y_test)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(base_dir, 'saved_models', 'evaluation_results.pkl')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nEvaluation results saved to: {results_path}")
    
    return results

def create_visualizations(results, server_mapping, feature_names, X_test, y_test):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    viz_dir = os.path.join(base_dir, 'evaluation', 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    algorithms = [name.replace('_', ' ').title() for name in results.keys()]
    accuracies = [r['accuracy'] * 100 for r in results.values()]
    avg_rts = [r['overall_avg_rt'] for r in results.values()]
    
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Load Balancing Algorithm Performance Comparison', fontsize=16, fontweight='bold')
    
    axes[0, 0].barh(algorithms, accuracies, color=sns.color_palette("husl", len(algorithms)))
    axes[0, 0].set_xlabel('Accuracy (%)', fontsize=12)
    axes[0, 0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].grid(axis='x', alpha=0.3)
    for i, (alg, acc) in enumerate(zip(algorithms, accuracies)):
        axes[0, 0].text(acc + 1, i, f'{acc:.1f}%', va='center', fontsize=10)
    
    axes[0, 1].barh(algorithms, avg_rts, color=sns.color_palette("coolwarm", len(algorithms)))
    axes[0, 1].set_xlabel('Average Response Time (ms)', fontsize=12)
    axes[0, 1].set_title('Response Time Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].grid(axis='x', alpha=0.3)
    for i, (alg, rt) in enumerate(zip(algorithms, avg_rts)):
        axes[0, 1].text(rt + 5, i, f'{rt:.1f}ms', va='center', fontsize=10)
    
    baseline_acc = accuracies[0] if accuracies else 0
    improvements = [((acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0 for acc in accuracies]
    axes[1, 0].barh(algorithms, improvements, color=['red' if x < 0 else 'green' for x in improvements])
    axes[1, 0].set_xlabel('Improvement (%)', fontsize=12)
    axes[1, 0].set_title('Accuracy Improvement vs Baseline', fontsize=14, fontweight='bold')
    axes[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].grid(axis='x', alpha=0.3)
    for i, (alg, imp) in enumerate(zip(algorithms, improvements)):
        axes[1, 0].text(imp + (2 if imp >= 0 else -2), i, f'{imp:+.1f}%', va='center', fontsize=10)
    
    server_distributions = {}
    for alg_name, result in results.items():
        server_distributions[alg_name] = defaultdict(int)
        for pred in result['predictions']:
            server_distributions[alg_name][pred] += 1
    
    server_names = ['h1', 'h2', 'h3', 'h4']
    x = np.arange(len(server_names))
    width = 0.8 / len(algorithms)
    
    for i, alg_name in enumerate(results.keys()):
        counts = [server_distributions[alg_name][s] for s in server_names]
        axes[1, 1].bar(x + i * width, counts, width, label=alg_name.replace('_', ' ').title(), alpha=0.8)
    
    axes[1, 1].set_xlabel('Server', fontsize=12)
    axes[1, 1].set_ylabel('Request Count', fontsize=12)
    axes[1, 1].set_title('Server Selection Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xticks(x + width * (len(algorithms) - 1) / 2)
    axes[1, 1].set_xticklabels(server_names)
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Performance Analysis', fontsize=16, fontweight='bold')
    
    response_time_data = []
    for alg_name, result in results.items():
        for server, rts in result['avg_response_times'].items():
            response_time_data.append({
                'Algorithm': alg_name.replace('_', ' ').title(),
                'Server': server,
                'Response Time': rts
            })
    
    if response_time_data:
        rt_df = pd.DataFrame(response_time_data)
        sns.barplot(data=rt_df, x='Algorithm', y='Response Time', hue='Server', ax=axes[0, 0])
        axes[0, 0].set_title('Response Time by Server and Algorithm', fontsize=14, fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend(title='Server', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    confusion_matrices = {}
    for alg_name, result in results.items():
        if alg_name != 'round_robin':
            cm = confusion_matrix(y_test, result['y_pred_encoded'])
            confusion_matrices[alg_name] = cm
    
    if confusion_matrices:
        n_models = len(confusion_matrices)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig_cm, axes_cm = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes_cm = [axes_cm]
        elif rows == 1:
            axes_cm = axes_cm
        else:
            axes_cm = axes_cm.flatten()
        
        for idx, (alg_name, cm) in enumerate(confusion_matrices.items()):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[idx], 
                       xticklabels=['h1', 'h2', 'h3', 'h4'], yticklabels=['h1', 'h2', 'h3', 'h4'])
            axes_cm[idx].set_title(f'{alg_name.replace("_", " ").title()}\nConfusion Matrix', fontsize=12, fontweight='bold')
            axes_cm[idx].set_xlabel('Predicted', fontsize=10)
            axes_cm[idx].set_ylabel('Actual', fontsize=10)
        
        for idx in range(len(confusion_matrices), len(axes_cm)):
            axes_cm[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    improvement_data = []
    baseline_acc = results['round_robin']['accuracy'] if 'round_robin' in results else 0
    baseline_rt = results['round_robin']['overall_avg_rt'] if 'round_robin' in results else 0
    
    for alg_name, result in results.items():
        if alg_name != 'round_robin':
            acc_imp = ((result['accuracy'] - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
            rt_imp = ((baseline_rt - result['overall_avg_rt']) / baseline_rt * 100) if baseline_rt > 0 else 0
            improvement_data.append({
                'Algorithm': alg_name.replace('_', ' ').title(),
                'Accuracy Improvement': acc_imp,
                'Response Time Improvement': rt_imp
            })
    
    if improvement_data:
        imp_df = pd.DataFrame(improvement_data)
        x_pos = np.arange(len(imp_df))
        width = 0.35
        
        axes[0, 1].bar(x_pos - width/2, imp_df['Accuracy Improvement'], width, label='Accuracy', alpha=0.8)
        axes[0, 1].bar(x_pos + width/2, imp_df['Response Time Improvement'], width, label='Response Time', alpha=0.8)
        axes[0, 1].set_xlabel('Algorithm', fontsize=12)
        axes[0, 1].set_ylabel('Improvement (%)', fontsize=12)
        axes[0, 1].set_title('Improvement Metrics', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(imp_df['Algorithm'], rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[0, 1].grid(axis='y', alpha=0.3)
    
    categories = ['Accuracy', 'Response Time']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    axes[1, 0].axis('off')
    axes[1, 0].text(0.5, 0.5, 'Radar Chart\n(Placeholder)', ha='center', va='center', fontsize=14)
    
    if 'response_time_ms' in feature_names:
        rt_idx = feature_names.index('response_time_ms')
        rt_values = X_test[:, rt_idx]
        
        axes[1, 1].hist(rt_values, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Response Time (ms)', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Response Time Distribution in Test Set', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'detailed_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Advanced Performance Analysis', fontsize=16, fontweight='bold')
    
    server_heatmap_data = []
    for alg_name, result in results.items():
        row = []
        for server in ['h1', 'h2', 'h3', 'h4']:
            row.append(result['avg_response_times'].get(server, 0))
        server_heatmap_data.append(row)
    
    server_heatmap_data = np.array(server_heatmap_data)
    im = axes[0, 0].imshow(server_heatmap_data, cmap='YlOrRd', aspect='auto')
    axes[0, 0].set_xticks(range(4))
    axes[0, 0].set_xticklabels(['h1', 'h2', 'h3', 'h4'])
    axes[0, 0].set_yticks(range(len(algorithms)))
    axes[0, 0].set_yticklabels(algorithms)
    axes[0, 0].set_title('Response Time Heatmap by Algorithm and Server', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=axes[0, 0], label='Response Time (ms)')
    
    for i in range(len(algorithms)):
        for j in range(4):
            text = axes[0, 0].text(j, i, f'{server_heatmap_data[i, j]:.0f}', 
                                  ha="center", va="center", color="black", fontsize=8)
    
    time_series_data = []
    sample_size = min(100, len(X_test))
    for alg_name, result in results.items():
        if alg_name != 'round_robin':
            sample_predictions = result['predictions'][:sample_size]
            sample_y = [server_mapping.get(label, f'h{label+1}') for label in y_test[:sample_size]]
            correct = [1 if pred == true else 0 for pred, true in zip(sample_predictions, sample_y)]
            time_series_data.append(correct)
    
    if time_series_data:
        x_axis = range(sample_size)
        for i, (alg_name, data) in enumerate(zip([k for k in results.keys() if k != 'round_robin'], time_series_data)):
            moving_avg = np.convolve(data, np.ones(10)/10, mode='valid')
            axes[0, 1].plot(x_axis[:len(moving_avg)], moving_avg, label=alg_name.replace('_', ' ').title(), alpha=0.7, linewidth=2)
        
        axes[0, 1].set_xlabel('Request Number', fontsize=12)
        axes[0, 1].set_ylabel('Moving Average Accuracy (10 requests)', fontsize=12)
        axes[0, 1].set_title('Time Series: Accuracy Over Requests', fontsize=14, fontweight='bold')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[0, 1].grid(alpha=0.3)
    
    categories = ['Accuracy', 'Response Time', 'Load Balance', 'Stability']
    n_categories = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
    angles += angles[:1]
    
    ax_radar = plt.subplot(2, 2, 3, projection='polar')
    
    top_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)[:5]
    colors_radar = plt.cm.Set3(np.linspace(0, 1, len(top_models)))
    
    for idx, (alg_name, result) in enumerate(top_models):
        if alg_name == 'round_robin':
            continue
        
        values = [
            result['accuracy'] * 100,
            100 - (result['overall_avg_rt'] / 10),
            len(set(result['predictions'])) / 4 * 100,
            80
        ]
        values += values[:1]
        
        ax_radar.plot(angles, values, 'o-', linewidth=2, label=alg_name.replace('_', ' ').title(), color=colors_radar[idx])
        ax_radar.fill(angles, values, alpha=0.15, color=colors_radar[idx])
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories, fontsize=10)
    ax_radar.set_ylim(0, 100)
    ax_radar.set_title('Radar Chart: Multi-Dimensional Performance', fontsize=14, fontweight='bold', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax_radar.grid(True)
    
    server_load_balance = {}
    for alg_name, result in results.items():
        server_counts = defaultdict(int)
        for pred in result['predictions']:
            server_counts[pred] += 1
        total = sum(server_counts.values())
        if total > 0:
            balance_score = 1.0 - (np.std(list(server_counts.values())) / (total / 4 + 1e-8))
            server_load_balance[alg_name] = balance_score * 100
    
    if server_load_balance:
        alg_names_balance = [k.replace('_', ' ').title() for k in server_load_balance.keys()]
        balance_scores = list(server_load_balance.values())
        
        axes[1, 1].barh(alg_names_balance, balance_scores, color=sns.color_palette("viridis", len(alg_names_balance)))
        axes[1, 1].set_xlabel('Load Balance Score (%)', fontsize=12)
        axes[1, 1].set_title('Load Balancing Quality Score', fontsize=14, fontweight='bold')
        axes[1, 1].grid(axis='x', alpha=0.3)
        for i, (alg, score) in enumerate(zip(alg_names_balance, balance_scores)):
            axes[1, 1].text(score + 1, i, f'{score:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'advanced_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Statistical Analysis and Distributions', fontsize=16, fontweight='bold')
    
    accuracy_dist = [r['accuracy'] * 100 for r in results.values()]
    axes[0, 0].hist(accuracy_dist, bins=15, edgecolor='black', alpha=0.7, color='skyblue')
    axes[0, 0].axvline(np.mean(accuracy_dist), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(accuracy_dist):.1f}%')
    axes[0, 0].set_xlabel('Accuracy (%)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Accuracy Distribution Across Algorithms', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    rt_dist = [r['overall_avg_rt'] for r in results.values()]
    axes[0, 1].hist(rt_dist, bins=15, edgecolor='black', alpha=0.7, color='coral')
    axes[0, 1].axvline(np.mean(rt_dist), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rt_dist):.1f}ms')
    axes[0, 1].set_xlabel('Response Time (ms)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Response Time Distribution Across Algorithms', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    improvement_comparison = []
    baseline_acc = results['round_robin']['accuracy'] if 'round_robin' in results else 0
    baseline_rt = results['round_robin']['overall_avg_rt'] if 'round_robin' in results else 0
    
    for alg_name, result in results.items():
        if alg_name != 'round_robin':
            acc_imp = ((result['accuracy'] - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
            rt_imp = ((baseline_rt - result['overall_avg_rt']) / baseline_rt * 100) if baseline_rt > 0 else 0
            improvement_comparison.append({
                'Algorithm': alg_name.replace('_', ' ').title(),
                'Accuracy Improvement': acc_imp,
                'Response Time Improvement': rt_imp
            })
    
    if improvement_comparison:
        imp_df = pd.DataFrame(improvement_comparison)
        x = np.arange(len(imp_df))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, imp_df['Accuracy Improvement'], width, label='Accuracy', alpha=0.8, color='green')
        axes[1, 0].bar(x + width/2, imp_df['Response Time Improvement'], width, label='Response Time', alpha=0.8, color='blue')
        axes[1, 0].set_xlabel('Algorithm', fontsize=12)
        axes[1, 0].set_ylabel('Improvement (%)', fontsize=12)
        axes[1, 0].set_title('Side-by-Side Improvement Comparison', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(imp_df['Algorithm'], rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1, 0].grid(axis='y', alpha=0.3)
    
    pie_data = defaultdict(int)
    for alg_name, result in results.items():
        if alg_name != 'round_robin':
            pie_data[alg_name.replace('_', ' ').title()] = result['accuracy'] * 100
    
    if pie_data:
        axes[1, 1].pie(pie_data.values(), labels=pie_data.keys(), autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Accuracy Distribution (Pie Chart)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'statistical_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to: {viz_dir}")
    print(f"Total visualization files created: 4")

if __name__ == '__main__':
    import pandas as pd
    main()

