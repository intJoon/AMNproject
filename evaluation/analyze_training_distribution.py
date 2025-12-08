import pickle
import os
import numpy as np
from collections import Counter

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pkl_path = os.path.join(base_dir, 'saved_models', 'preprocessed_data.pkl')

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

y_train = data['y_train']
server_mapping = data['server_mapping']

servers = [server_mapping.get(y, f'h{y+1}') for y in y_train]

print('Training Data Server Distribution:')
dist = Counter(servers)
print(dist)
print('\nPercentage:')
total = len(servers)
for s in ['h1', 'h2', 'h3', 'h4']:
    count = dist.get(s, 0)
    print(f'{s}: {count} ({count/total*100:.2f}%)')

print(f'\nTotal samples: {total}')

