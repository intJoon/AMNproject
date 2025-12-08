import pandas as pd
import sys
import os
import json
import numpy as np
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.preprocess_data import create_features, create_labels, prepare_model_features
from models.random_forest_model import RandomForestLoadBalancer

test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         'dataset', '[for testing]training_data_300samples_20251207_121637.json')

with open(test_file, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = create_features(df)
df = create_labels(df)
X, y, feature_names = prepare_model_features(df, exclude_server_column=True)

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          'saved_models', 'random_forest.pkl')
model = RandomForestLoadBalancer.load(model_path)
model.server_mapping = {0: 'h1', 1: 'h2', 2: 'h3', 3: 'h4'}

server_mapping = {0: 'h1', 1: 'h2', 2: 'h3', 3: 'h4'}
predictions = []
current_loads = defaultdict(int)

for i, features in enumerate(X):
    pred_server = model.predict_server(features, dict(current_loads), server_mapping)
    predictions.append(pred_server)
    current_loads[pred_server] += 1

true_servers = [server_mapping.get(label, f'h{label+1}') for label in y]

comparison = pd.DataFrame({
    'Actual_Optimal': true_servers,
    'Predicted': predictions
})

comparison['Correct'] = comparison['Actual_Optimal'] == comparison['Predicted']
accuracy = comparison['Correct'].mean()

print("=" * 80)
print("Load-Aware Random Forest Test Results")
print("=" * 80)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nServer Distribution:")
print(comparison['Predicted'].value_counts())
print(f"\nPercentage:")
print(comparison['Predicted'].value_counts(normalize=True) * 100)
print(f"\nAll servers selected: {sorted(set(comparison['Predicted'].unique()))}")
print(f"\nFinal Load Distribution:")
for server in ['h1', 'h2', 'h3', 'h4']:
    print(f"  {server}: {current_loads[server]} requests")

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'load_aware_test_results.csv')
comparison.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")

