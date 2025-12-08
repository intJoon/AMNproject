import json
import pickle
import os
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.model_selection import train_test_split

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def create_features(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['server_encoded'] = df['server'].astype('category').cat.codes
    
    server_stats = df.groupby('server').agg({
        'response_time_ms': ['mean', 'std', 'min', 'max', 'count'],
        'success': 'mean'
    }).reset_index()
    
    server_stats.columns = ['server', 'mean_rt', 'std_rt', 'min_rt', 'max_rt', 'count', 'success_rate']
    
    df = df.merge(server_stats, on='server', how='left')
    
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['second'] = df['timestamp'].dt.second
    
    window_size = 10
    for server in ['h1', 'h2', 'h3', 'h4']:
        server_data = df[df['server'] == server].copy()
        server_data = server_data.sort_values('timestamp')
        server_data[f'{server}_rolling_mean'] = server_data['response_time_ms'].rolling(window=window_size, min_periods=1).mean()
        server_data[f'{server}_rolling_std'] = server_data['response_time_ms'].rolling(window=window_size, min_periods=1).std().fillna(0)
        df.loc[df['server'] == server, f'{server}_rolling_mean'] = server_data[f'{server}_rolling_mean']
        df.loc[df['server'] == server, f'{server}_rolling_std'] = server_data[f'{server}_rolling_std']
    
    current_time_features = []
    for idx, row in df.iterrows():
        current_time = row['timestamp']
        time_window = pd.Timedelta(seconds=5)
        
        recent_data = df[
            (df['timestamp'] <= current_time) & 
            (df['timestamp'] > current_time - time_window)
        ]
        
        row_features = {}
        for server in ['h1', 'h2', 'h3', 'h4']:
            server_recent = recent_data[recent_data['server'] == server]
            if len(server_recent) > 0:
                row_features[f'{server}_recent_mean'] = server_recent['response_time_ms'].mean()
                row_features[f'{server}_recent_count'] = len(server_recent)
                row_features[f'{server}_recent_success_rate'] = server_recent['success'].mean()
            else:
                row_features[f'{server}_recent_mean'] = np.nan
                row_features[f'{server}_recent_count'] = 0
                row_features[f'{server}_recent_success_rate'] = 1.0
        
        current_time_features.append(row_features)
    
    time_features_df = pd.DataFrame(current_time_features)
    df = pd.concat([df.reset_index(drop=True), time_features_df.reset_index(drop=True)], axis=1)
    
    df = df.bfill().ffill().fillna(0)
    
    return df

def create_labels(df):
    labels = []
    for idx, row in df.iterrows():
        current_time = row['timestamp']
        time_window = pd.Timedelta(seconds=2)
        
        future_data = df[
            (df['timestamp'] > current_time) & 
            (df['timestamp'] <= current_time + time_window)
        ]
        
        if len(future_data) > 0:
            server_performance = []
            for server in ['h1', 'h2', 'h3', 'h4']:
                server_data = future_data[future_data['server'] == server]
                if len(server_data) > 0:
                    avg_response_time = server_data['response_time_ms'].mean()
                    success_rate = server_data['success'].mean()
                    score = avg_response_time * (2 - success_rate)
                else:
                    score = 999999
                server_performance.append(score)
            
            best_server_idx = np.argmin(server_performance)
            best_server = ['h1', 'h2', 'h3', 'h4'][best_server_idx]
        else:
            current_server_perf = []
            for server in ['h1', 'h2', 'h3', 'h4']:
                server_data = df[df['server'] == server]
                if len(server_data) > 0:
                    historical_data = server_data[server_data['timestamp'] <= current_time]
                    if len(historical_data) > 0:
                        avg_response_time = historical_data['response_time_ms'].mean()
                        success_rate = historical_data['success'].mean()
                        score = avg_response_time * (2 - success_rate)
                    else:
                        score = 999999
                else:
                    score = 999999
                current_server_perf.append(score)
            
            best_server_idx = np.argmin(current_server_perf)
            best_server = ['h1', 'h2', 'h3', 'h4'][best_server_idx]
        
        labels.append(best_server)
    
    df['optimal_server'] = labels
    df['optimal_server_encoded'] = df['optimal_server'].astype('category').cat.codes
    
    return df

def prepare_model_features(df, exclude_server_column=False):
    feature_cols = [
        'server_encoded', 'response_time_ms', 'success',
        'mean_rt', 'std_rt', 'min_rt', 'max_rt', 'count', 'success_rate',
        'hour', 'minute', 'second',
        'h1_recent_mean', 'h1_recent_count', 'h1_recent_success_rate',
        'h2_recent_mean', 'h2_recent_count', 'h2_recent_success_rate',
        'h3_recent_mean', 'h3_recent_count', 'h3_recent_success_rate',
        'h4_recent_mean', 'h4_recent_count', 'h4_recent_success_rate'
    ]
    
    if exclude_server_column:
        feature_cols = [col for col in feature_cols if col != 'server_encoded']
    
    available_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_cols].values
    
    if 'optimal_server_encoded' in df.columns:
        y = df['optimal_server_encoded'].values
    else:
        y = None
    
    return X, y, available_cols

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    files = [
        'training_data_250samples_20251206_110938.json',
        'training_data_300samples_20251206_112558.json'
    ]
    
    all_data = []
    for file in files:
        file_path = os.path.join(dataset_dir, file)
        if os.path.exists(file_path):
            data = load_json_data(file_path)
            all_data.extend(data)
            print(f"Loaded {len(data)} samples from {file}")
    
    print(f"Total samples: {len(all_data)}")
    
    df = pd.DataFrame(all_data)
    print(f"Data shape: {df.shape}")
    print(f"Servers: {df['server'].unique()}")
    print(f"\nServer statistics:")
    print(df.groupby('server')['response_time_ms'].describe())
    print(f"\nSuccess rates by server:")
    print(df.groupby('server')['success'].mean())
    
    df = create_features(df)
    print("\nFeatures created")
    
    df = create_labels(df)
    print("Labels created")
    
    print(f"\nOptimal server distribution:")
    print(df['optimal_server'].value_counts())
    
    X, y, feature_names = prepare_model_features(df, exclude_server_column=True)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Feature names: {feature_names}")
    print("Note: server_encoded excluded from features as per deployment requirements")
    
    if y is not None:
        unique_classes = np.unique(y)
        min_class_count = min([np.sum(y == cls) for cls in unique_classes])
        
        if min_class_count >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            unique_train_classes = np.unique(y_train)
            min_train_class_count = min([np.sum(y_train == cls) for cls in unique_train_classes])
            
            if min_train_class_count >= 2:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                )
            else:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
    else:
        X_train, X_val, X_test = None, None, None
        y_train, y_val, y_test = None, None, None
    
    saved_models_dir = os.path.join(base_dir, 'saved_models')
    os.makedirs(saved_models_dir, exist_ok=True)
    
    preprocessed_data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': feature_names,
        'df': df,
        'server_mapping': {i: server for i, server in enumerate(['h1', 'h2', 'h3', 'h4'])}
    }
    
    pkl_path = os.path.join(saved_models_dir, 'preprocessed_data.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    print(f"\nPreprocessed data saved to: {pkl_path}")
    
    return preprocessed_data

if __name__ == '__main__':
    preprocessed_data = main()
