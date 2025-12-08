import pandas as pd
import sys
import os
import json
import numpy as np

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

predictions = model.predict(X, use_probabilistic=True)

server_mapping = {0: 'h1', 1: 'h2', 2: 'h3', 3: 'h4'}
predicted_servers = [server_mapping.get(p, f'h{p+1}') for p in predictions]

comparison = pd.DataFrame({
    'Actual_Optimal': df['optimal_server'],
    'Predicted': predicted_servers
})

comparison['Correct'] = comparison['Actual_Optimal'] == comparison['Predicted']
accuracy = comparison['Correct'].mean()

print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nServer Distribution:")
print(comparison['Predicted'].value_counts())
print(f"\nPercentage:")
print(comparison['Predicted'].value_counts(normalize=True) * 100)
print(f"\nAll servers selected: {sorted(set(comparison['Predicted'].unique()))}")

