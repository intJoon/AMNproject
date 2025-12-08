import pandas as pd
import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.preprocess_data import create_features, create_labels, prepare_model_features
from models.random_forest_model import RandomForestLoadBalancer

def load_model(model_path):
    if os.path.exists(model_path):
        return RandomForestLoadBalancer.load(model_path)
    return None

def evaluate_model(model, X, y, feature_names, server_mapping, use_load_aware=False):
    predictions = []
    current_loads = defaultdict(int)
    
    for i, features in enumerate(X):
        if use_load_aware:
            pred_server = model.predict_server(features, dict(current_loads), server_mapping)
        else:
            pred_server = model.predict_server(features, None, server_mapping)
        predictions.append(pred_server)
        if use_load_aware:
            current_loads[pred_server] += 1
    
    true_servers = [server_mapping.get(label, f'h{label+1}') for label in y]
    
    comparison = pd.DataFrame({
        'Actual_Optimal': true_servers,
        'Predicted': predictions
    })
    
    comparison['Correct'] = comparison['Actual_Optimal'] == comparison['Predicted']
    accuracy = comparison['Correct'].mean()
    
    server_dist = comparison['Predicted'].value_counts()
    server_percent = comparison['Predicted'].value_counts(normalize=True) * 100
    
    return {
        'accuracy': accuracy,
        'predictions': predictions,
        'server_distribution': server_dist.to_dict(),
        'server_percentage': server_percent.to_dict(),
        'all_servers': sorted(set(predictions)),
        'final_loads': dict(current_loads) if use_load_aware else None
    }

test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         'dataset', '[for testing]training_data_300samples_20251207_121637.json')

print("Loading test data...")
with open(test_file, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = create_features(df)
df = create_labels(df)
X_with_server, y, feature_names_with_server = prepare_model_features(df, exclude_server_column=False)
X_without_server, y, feature_names_without_server = prepare_model_features(df, exclude_server_column=True)

server_mapping = {0: 'h1', 1: 'h2', 2: 'h3', 3: 'h4'}

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
before_model_path = os.path.join(base_dir, 'saved_models', 'random_forest_before.pkl')
current_model_path = os.path.join(base_dir, 'saved_models', 'random_forest.pkl')

print("\nLoading models...")
before_model = load_model(before_model_path)
current_model = load_model(current_model_path)

if before_model:
    before_model.server_mapping = server_mapping
    print("  ✓ Loaded random_forest_before.pkl")
else:
    print("  ✗ random_forest_before.pkl not found, using current model as baseline")

current_model.server_mapping = server_mapping
print("  ✓ Loaded random_forest.pkl (current)")

print("\nEvaluating models...")
if before_model:
    before_results = evaluate_model(before_model, X_with_server, y, feature_names_with_server, server_mapping, use_load_aware=False)
    print("  ✓ Evaluated before model")
else:
    before_results = evaluate_model(current_model, X_without_server, y, feature_names_without_server, server_mapping, use_load_aware=False)
    print("  ✓ Evaluated current model (without load awareness)")

current_results = evaluate_model(current_model, X_without_server, y, feature_names_without_server, server_mapping, use_load_aware=True)
print("  ✓ Evaluated current model (with load awareness)")

print("\n" + "=" * 80)
print("COMPARISON RESULTS")
print("=" * 80)

if before_model:
    print("\nBEFORE (random_forest_before.pkl):")
else:
    print("\nCURRENT (without load awareness):")
print(f"  Accuracy: {before_results['accuracy']:.4f} ({before_results['accuracy']*100:.2f}%)")
print(f"  Server Distribution:")
for server in ['h1', 'h2', 'h3', 'h4']:
    count = before_results['server_distribution'].get(server, 0)
    pct = before_results['server_percentage'].get(server, 0)
    print(f"    {server}: {count} ({pct:.2f}%)")
print(f"  Servers used: {before_results['all_servers']}")

print("\nCURRENT (with load awareness):")
print(f"  Accuracy: {current_results['accuracy']:.4f} ({current_results['accuracy']*100:.2f}%)")
print(f"  Server Distribution:")
for server in ['h1', 'h2', 'h3', 'h4']:
    count = current_results['server_distribution'].get(server, 0)
    pct = current_results['server_percentage'].get(server, 0)
    print(f"    {server}: {count} ({pct:.2f}%)")
print(f"  Servers used: {current_results['all_servers']}")
if current_results['final_loads']:
    print(f"  Final Load Distribution:")
    for server in ['h1', 'h2', 'h3', 'h4']:
        print(f"    {server}: {current_results['final_loads'].get(server, 0)} requests")

print("\n" + "=" * 80)
print("Creating visualization...")

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('white')

title_y = 0.97
fig.text(0.5, title_y, 'Random Forest Load Balancer: Before vs After Comparison', 
         ha='center', va='top', fontsize=24, fontweight='bold')
fig.text(0.5, title_y - 0.03, 'Load Balancing Performance Analysis', 
         ha='center', va='top', fontsize=16, style='italic', color='gray')

servers = ['h1', 'h2', 'h3', 'h4']
before_counts = [before_results['server_distribution'].get(s, 0) for s in servers]
current_counts = [current_results['server_distribution'].get(s, 0) for s in servers]
before_pcts = [before_results['server_percentage'].get(s, 0) for s in servers]
current_pcts = [current_results['server_percentage'].get(s, 0) for s in servers]

x = np.arange(len(servers))
width = 0.35

ax1 = fig.add_axes([0.10, 0.55, 0.38, 0.35])
bars1 = ax1.bar(x - width/2, before_counts, width, label='Before', color='#FF6B6B', edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x + width/2, current_counts, width, label='After (Load-Aware)', color='#4ECDC4', edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Server', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Requests', fontsize=12, fontweight='bold')
ax1.set_title('Server Request Distribution', fontsize=14, fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(servers)
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2 = fig.add_axes([0.57, 0.55, 0.38, 0.35])
bars3 = ax2.bar(x - width/2, before_pcts, width, label='Before', color='#FF6B6B', edgecolor='black', linewidth=1.5)
bars4 = ax2.bar(x + width/2, current_pcts, width, label='After (Load-Aware)', color='#4ECDC4', edgecolor='black', linewidth=1.5)

ax2.set_xlabel('Server', fontsize=12, fontweight='bold')
ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title('Server Request Percentage', fontsize=14, fontweight='bold', pad=10)
ax2.set_xticks(x)
ax2.set_xticklabels(servers)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim(0, max(max(before_pcts), max(current_pcts)) * 1.2)

for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax3 = fig.add_axes([0.10, 0.10, 0.38, 0.35])
metrics = ['Accuracy', 'Servers Used']
before_metrics = [before_results['accuracy'] * 100, len(before_results['all_servers'])]
current_metrics = [current_results['accuracy'] * 100, len(current_results['all_servers'])]

x_metrics = np.arange(len(metrics))
bars5 = ax3.bar(x_metrics - width/2, before_metrics, width, label='Before', color='#FF6B6B', edgecolor='black', linewidth=1.5)
bars6 = ax3.bar(x_metrics + width/2, current_metrics, width, label='After (Load-Aware)', color='#4ECDC4', edgecolor='black', linewidth=1.5)

ax3.set_xlabel('Metric', fontsize=12, fontweight='bold')
ax3.set_ylabel('Value', fontsize=12, fontweight='bold')
ax3.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=10)
ax3.set_xticks(x_metrics)
ax3.set_xticklabels(metrics)
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

for i, (b1, b2) in enumerate(zip(bars5, bars6)):
    h1, h2 = b1.get_height(), b2.get_height()
    if i == 0:
        ax3.text(b1.get_x() + b1.get_width()/2., h1, f'{h1:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax3.text(b2.get_x() + b2.get_width()/2., h2, f'{h2:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax3.text(b1.get_x() + b1.get_width()/2., h1, f'{int(h1)}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax3.text(b2.get_x() + b2.get_width()/2., h2, f'{int(h2)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax4 = fig.add_axes([0.57, 0.10, 0.38, 0.35])
improvements = []
for server in servers:
    before_val = before_pcts[servers.index(server)]
    current_val = current_pcts[servers.index(server)]
    if before_val == 0:
        improvement = 100 if current_val > 0 else 0
    else:
        improvement = ((current_val - before_val) / before_val) * 100 if before_val > 0 else 0
    improvements.append(improvement)

colors_imp = ['#4ECDC4' if x >= 0 else '#FF6B6B' for x in improvements]
bars7 = ax4.bar(x, improvements, width=0.6, color=colors_imp, edgecolor='black', linewidth=1.5)
ax4.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax4.set_xlabel('Server', fontsize=12, fontweight='bold')
ax4.set_ylabel('Change (%)', fontsize=12, fontweight='bold')
ax4.set_title('Server Usage Change', fontsize=14, fontweight='bold', pad=10)
ax4.set_xticks(x)
ax4.set_xticklabels(servers)
ax4.grid(axis='y', alpha=0.3, linestyle='--')

for bar, imp in zip(bars7, improvements):
    if imp != 0:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{imp:+.1f}%', ha='center', va='bottom' if imp >= 0 else 'top', fontsize=9, fontweight='bold')

info_y = 0.02
info_text = (
    f"Before: Accuracy {before_results['accuracy']*100:.1f}%, Servers: {len(before_results['all_servers'])} | "
    f"After: Accuracy {current_results['accuracy']*100:.1f}%, Servers: {len(current_results['all_servers'])} | "
    f"Load Balancing: {'✓ Improved' if len(current_results['all_servers']) > len(before_results['all_servers']) else '= Same'}"
)

fig.text(0.5, info_y, info_text, ha='center', va='bottom', fontsize=11,
         bbox=dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.85, 
                  edgecolor='black', linewidth=1.5))

output_path = os.path.join(base_dir, 'evaluation', 'random_forest_comparison.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.3)
plt.close()

print(f"  ✓ Visualization saved to: {output_path}")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)

