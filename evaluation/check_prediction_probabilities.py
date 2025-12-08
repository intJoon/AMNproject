import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.random_forest_model import RandomForestLoadBalancer
from data_processing.preprocess_data import create_features, create_labels, prepare_model_features

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

proba = model.predict_proba(X)

print("Prediction Probabilities Analysis:")
print(f"Shape: {proba.shape}")
print(f"\nFirst 10 samples probabilities:")
for i in range(min(10, len(proba))):
    print(f"Sample {i}: {proba[i]}")

print(f"\nAverage probabilities per class:")
avg_proba = np.mean(proba, axis=0)
for i, prob in enumerate(avg_proba):
    server = ['h1', 'h2', 'h3', 'h4'][i]
    print(f"{server}: {prob:.4f}")

print(f"\nSamples where h2 or h4 has > 0.1 probability:")
h2_h4_high_prob = []
for i, p in enumerate(proba):
    if p[1] > 0.1 or p[3] > 0.1:
        h2_h4_high_prob.append(i)
        print(f"Sample {i}: h1={p[0]:.3f}, h2={p[1]:.3f}, h3={p[2]:.3f}, h4={p[3]:.3f}")

print(f"\nTotal samples with h2/h4 probability > 0.1: {len(h2_h4_high_prob)}")

