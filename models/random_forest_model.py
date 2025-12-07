import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import math

class RandomForestLoadBalancer:
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
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
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def predict_server(self, features, server_mapping=None):
        if server_mapping is None:
            server_mapping = self.server_mapping
        
        if server_mapping is None:
            server_mapping = {0: 'h1', 1: 'h2', 2: 'h3', 3: 'h4'}
        
        X = np.array(features).reshape(1, -1)
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

