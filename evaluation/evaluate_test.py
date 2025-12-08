import pandas as pd
import sys
import os
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.preprocess_data import create_features, create_labels, prepare_model_features
from models.random_forest_model import RandomForestLoadBalancer

def main():
    # Load test data
    test_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                             'dataset', '[for testing]training_data_300samples_20251207_121637.json')
    
    print(f"Loading test data from: {test_file}")
    with open(test_file, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"Loaded {len(df)} samples")

    # Preprocess
    print("Preprocessing data...")
    df = create_features(df)
    df = create_labels(df)
    
    # Remove server column from features for test evaluation (as per instructions)
    X, y, feature_names = prepare_model_features(df, exclude_server_column=True)
    
    print(f"Features shape: {X.shape} (server column removed as per instructions)")
    print(f"Feature names: {feature_names[:5]}... (total {len(feature_names)} features)")

    # Load model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'saved_models', 'random_forest.pkl')
    print(f"Loading model from: {model_path}")
    model = RandomForestLoadBalancer.load(model_path)

    # Predict
    print("Predicting...")
    predictions = model.predict(X)

    # Map predictions
    server_mapping = {0: 'h1', 1: 'h2', 2: 'h3', 3: 'h4'}
    predicted_servers = [server_mapping.get(p, f'h{p+1}') for p in predictions]

    # Create comparison dataframe (removing 'server' column as requested for report)
    comparison = pd.DataFrame({
        'Actual_Optimal': df['optimal_server'],
        'Predicted': predicted_servers
    })

    # Add correctness check
    comparison['Correct'] = comparison['Actual_Optimal'] == comparison['Predicted']
    
    accuracy = comparison['Correct'].mean()
    print(f"Accuracy on Test Data: {accuracy:.4f}")

    # Save to CSV
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_evaluation_report.csv')
    comparison.to_csv(output_path, index=False)
    print(f"Report saved to: {output_path}")
    
    # Display first few rows
    print("\nFirst 20 predictions:")
    print(comparison.head(20))

if __name__ == '__main__':
    main()

