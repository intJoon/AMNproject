import numpy as np
from collections import defaultdict

class RoundRobinLoadBalancer:
    def __init__(self):
        self.current_index = 0
        self.servers = ['h1', 'h2', 'h3', 'h4']
    
    def get_next_server(self):
        server = self.servers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.servers)
        return server
    
    def predict_server(self, features=None):
        return self.get_next_server()
    
    def predict(self, X):
        predictions = []
        for _ in range(len(X)):
            predictions.append(self.get_next_server())
        return np.array(predictions)
    
    def reset(self):
        self.current_index = 0

