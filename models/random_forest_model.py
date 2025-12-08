import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import math

class RandomForestLoadBalancer:
    def __init__(self, n_estimators=100, max_depth=5, min_samples_split=15, min_samples_leaf=8, max_features='sqrt', random_state=42, class_weight='balanced'):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,
            class_weight=class_weight
        )
        self.server_mapping = None
        self.feature_names = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            print(f"Validation Accuracy: {accuracy:.4f}")
            print(f"\nClassification Report:\n{classification_report(y_val, y_pred)}")
            
        return self.model
    
    def predict(self, X, use_probabilistic=False, load_aware=False, current_loads=None):
        if load_aware and current_loads is not None:
            proba = self.model.predict_proba(X)
            predictions = []
            server_mapping = self.server_mapping if self.server_mapping else {0: 'h1', 1: 'h2', 2: 'h3', 3: 'h4'}
            servers = ['h1', 'h2', 'h3', 'h4']
            
            for p in proba:
                adjusted_proba = p.copy()
                for i, server in enumerate(servers):
                    load = current_loads.get(server, 0)
                    adjusted_proba[i] *= (1.0 / (1.0 + load * 0.1))
                
                adjusted_proba = adjusted_proba / adjusted_proba.sum()
                predictions.append(np.random.choice(len(adjusted_proba), p=adjusted_proba))
            return np.array(predictions)
        elif use_probabilistic:
            proba = self.model.predict_proba(X)
            predictions = []
            for p in proba:
                predictions.append(np.random.choice(len(p), p=p))
            return np.array(predictions)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def predict_server(self, features, current_loads=None, server_mapping=None):
        if server_mapping is None:
            server_mapping = self.server_mapping
        
        if server_mapping is None:
            server_mapping = {0: 'h1', 1: 'h2', 2: 'h3', 3: 'h4'}
        
        X = np.array(features).reshape(1, -1)
        
        if current_loads is not None and len(current_loads) > 0:
            proba = self.model.predict_proba(X)[0]
            servers = ['h1', 'h2', 'h3', 'h4']
            
            adjusted_proba = proba.copy()
            max_load = max(current_loads.values()) if current_loads.values() else 1
            
            for i, server in enumerate(servers):
                load = current_loads.get(server, 0)
                load_factor = 1.0 / (1.0 + load * 0.2)
                adjusted_proba[i] *= load_factor
            
            adjusted_proba = adjusted_proba / adjusted_proba.sum()
            
            if np.random.random() < 0.7:
                pred_class = np.argmax(adjusted_proba)
            else:
                pred_class = np.random.choice(len(adjusted_proba), p=adjusted_proba)
        else:
            pred_class = self.predict(X)[0]
        
        return server_mapping.get(pred_class, f'h{pred_class + 1}')
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'server_mapping': self.server_mapping,
                'feature_names': self.feature_names
            }, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.model = data['model']
        instance.server_mapping = data['server_mapping']
        instance.feature_names = data['feature_names']
        return instance

