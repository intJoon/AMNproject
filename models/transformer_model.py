import pickle
import os
import numpy as np
import math

class TransformerLoadBalancer:
    def __init__(self, d_model=24, n_heads=4, n_layers=2, seq_length=10, dropout=0.1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_length = seq_length
        self.dropout = dropout
        
        self.positional_encoding = None
        self.attention_weights = []
        self.encoder_layers = []
        self.output_projection = None
        self.feature_names = None
        self.server_mapping = None
        
    def positional_encoding_matrix(self, seq_len, d_model):
        pe = np.zeros((seq_len, d_model))
        position = np.arange(0, seq_len, dtype=np.float32).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2).astype(np.float32) * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.shape[-1]
        
        if Q.ndim == 3:
            K_transposed = K.swapaxes(-2, -1)
            scores = np.matmul(Q, K_transposed) / math.sqrt(d_k)
        else:
            scores = np.dot(Q, K.T) / math.sqrt(d_k)
        
        scores = np.asarray(scores, dtype=np.float64)
        
        if scores.ndim == 0:
            scores = np.array([[float(scores)]])
        elif scores.ndim == 1:
            scores = scores.reshape(1, -1)
        
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        if scores.ndim >= 2:
            max_scores = np.max(scores, axis=-1, keepdims=True)
            scores_diff = scores - max_scores
            scores_diff = np.asarray(scores_diff, dtype=np.float64)
            scores_diff = np.clip(scores_diff, -500, 500)
            exp_scores = np.exp(scores_diff)
            attention_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-8)
        else:
            max_score = float(np.max(scores))
            scores_array = np.asarray(scores, dtype=np.float64).flatten()
            scores_diff = scores_array - max_score
            scores_diff = np.clip(scores_diff, -500, 500)
            exp_scores = np.exp(scores_diff)
            attention_weights = exp_scores / (np.sum(exp_scores) + 1e-8)
            if attention_weights.ndim == 1:
                attention_weights = attention_weights.reshape(-1, 1)
        
        if Q.ndim == 3:
            output = np.matmul(attention_weights, V)
        else:
            if attention_weights.ndim == 2 and attention_weights.shape[0] == 1:
                output = np.dot(attention_weights[0], V)
            else:
                output = np.dot(attention_weights, V)
        
        return output, attention_weights
    
    def multi_head_attention(self, x, layer_idx=0):
        batch_size, seq_len, d_model = x.shape
        d_k = d_model // self.n_heads
        
        if len(self.encoder_layers) <= layer_idx:
            W_q = np.random.randn(d_model, d_model) * 0.1
            W_k = np.random.randn(d_model, d_model) * 0.1
            W_v = np.random.randn(d_model, d_model) * 0.1
            W_o = np.random.randn(d_model, d_model) * 0.1
            self.encoder_layers.append({'W_q': W_q, 'W_k': W_k, 'W_v': W_v, 'W_o': W_o})
        
        weights = self.encoder_layers[layer_idx]
        
        Q = np.dot(x, weights['W_q'])
        K = np.dot(x, weights['W_k'])
        V = np.dot(x, weights['W_v'])
        
        Q = Q.reshape(batch_size, seq_len, self.n_heads, d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, d_k).transpose(0, 2, 1, 3)
        
        Q = Q.reshape(batch_size * self.n_heads, seq_len, d_k)
        K = K.reshape(batch_size * self.n_heads, seq_len, d_k)
        V = V.reshape(batch_size * self.n_heads, seq_len, d_k)
        
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V)
        
        attn_output = attn_output.reshape(batch_size, self.n_heads, seq_len, d_k).transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch_size, seq_len, d_model)
        
        output = np.dot(attn_output, weights['W_o'])
        
        return output, attn_weights
    
    def feed_forward(self, x, layer_idx=0):
        d_model = x.shape[-1]
        d_ff = d_model * 4
        
        if len(self.encoder_layers) <= layer_idx:
            self.encoder_layers.append({})
        
        if 'ff_w1' not in self.encoder_layers[layer_idx]:
            self.encoder_layers[layer_idx]['ff_w1'] = np.random.randn(d_model, d_ff) * 0.1
            self.encoder_layers[layer_idx]['ff_w2'] = np.random.randn(d_ff, d_model) * 0.1
            self.encoder_layers[layer_idx]['ff_b1'] = np.zeros(d_ff)
            self.encoder_layers[layer_idx]['ff_b2'] = np.zeros(d_model)
        
        weights = self.encoder_layers[layer_idx]
        
        ff1 = np.dot(x, weights['ff_w1']) + weights['ff_b1']
        ff1 = np.maximum(0, ff1)
        ff1 = np.clip(ff1, -1e6, 1e6)
        ff2 = np.dot(ff1, weights['ff_w2']) + weights['ff_b2']
        ff2 = np.clip(ff2, -1e6, 1e6)
        
        return ff2
    
    def encoder_layer(self, x, layer_idx=0):
        attn_output, attn_weights = self.multi_head_attention(x, layer_idx)
        x = x + attn_output
        x = np.clip(x, -1e6, 1e6)
        
        ff_output = self.feed_forward(x, layer_idx)
        x = x + ff_output
        x = np.clip(x, -1e6, 1e6)
        
        return x, attn_weights
    
    def forward(self, x):
        if self.positional_encoding is None:
            self.positional_encoding = self.positional_encoding_matrix(self.seq_length, self.d_model)
        
        seq_len = x.shape[1]
        if seq_len < self.seq_length:
            padding = np.zeros((x.shape[0], self.seq_length - seq_len, x.shape[2]))
            x = np.concatenate([x, padding], axis=1)
        elif seq_len > self.seq_length:
            x = x[:, -self.seq_length:, :]
        
        x = x + self.positional_encoding[:x.shape[1], :]
        
        all_attn_weights = []
        for layer_idx in range(self.n_layers):
            x, attn_weights = self.encoder_layer(x, layer_idx)
            all_attn_weights.append(attn_weights)
        
        if self.output_projection is None:
            self.output_projection = np.asarray(np.random.randn(self.d_model, 4) * 0.1, dtype=np.float64)
        
        pooled = np.mean(x, axis=1)
        pooled = np.clip(pooled, -1e6, 1e6)
        output = np.dot(pooled, self.output_projection)
        output = np.clip(output, -1e6, 1e6)
        
        return output, all_attn_weights
    
    def prepare_sequence(self, features, history=None):
        if history is None:
            history = []
        
        history.append(features.copy())
        if len(history) > self.seq_length:
            history.pop(0)
        
        sequence_list = []
        for feat in history:
            if len(feat) < self.d_model:
                feat_padded = np.pad(feat, (0, self.d_model - len(feat)))
            else:
                feat_padded = feat[:self.d_model]
            sequence_list.append(feat_padded)
        
        while len(sequence_list) < self.seq_length:
            zero_feat = np.zeros(self.d_model)
            sequence_list.insert(0, zero_feat)
        
        sequence = np.array(sequence_list)
        
        return sequence.reshape(1, self.seq_length, self.d_model), history
    
    def predict_server(self, features, server_mapping=None, history=None):
        if server_mapping is None:
            server_mapping = self.server_mapping if self.server_mapping else {0: 'h1', 1: 'h2', 2: 'h3', 3: 'h4'}
        
        sequence, updated_history = self.prepare_sequence(features, history)
        output, _ = self.forward(sequence)
        
        pred_class = np.argmax(output[0])
        return server_mapping.get(pred_class, f'h{pred_class + 1}'), updated_history
    
    def train(self, X_train, y_train, X_val=None, y_val=None, server_mapping=None, feature_names=None, epochs=50):
        self.feature_names = feature_names if feature_names is not None else []
        self.server_mapping = server_mapping if server_mapping else {0: 'h1', 1: 'h2', 2: 'h3', 3: 'h4'}
        
        initial_learning_rate = 0.01
        history = []
        
        for epoch in range(epochs):
            learning_rate = initial_learning_rate * (0.98 ** (epoch // 5))
            learning_rate = max(learning_rate, initial_learning_rate * 0.2)
            total_loss = 0
            correct = 0
            
            for i in range(len(X_train)):
                features = X_train[i]
                true_label = y_train[i]
                
                sequence, history = self.prepare_sequence(features, history)
                output, _ = self.forward(sequence)
                
                output_logits = np.asarray(output[0], dtype=np.float64)
                if output_logits.ndim == 0:
                    output_logits = np.array([output_logits], dtype=np.float64)
                output_logits = output_logits.flatten()
                if len(output_logits) < 4:
                    output_logits = np.pad(output_logits, (0, 4 - len(output_logits)), 'constant')
                elif len(output_logits) > 4:
                    output_logits = output_logits[:4]
                
                max_logit = float(np.max(output_logits))
                diff = np.asarray(output_logits, dtype=np.float64) - max_logit
                diff = np.clip(diff, -500, 500)
                exp_logits = np.exp(diff)
                probs = exp_logits / (np.sum(exp_logits) + 1e-8)
                
                pred_class = np.argmax(probs)
                
                target = np.zeros(4, dtype=np.float64)
                target[true_label] = 1.0
                loss = -np.sum(target * np.log(probs + 1e-8))
                total_loss += loss
                
                if pred_class == true_label:
                    correct += 1
                
                if self.output_projection is not None:
                    gradient = np.asarray((probs - target) * learning_rate * 3.0, dtype=np.float64)
                    gradient = np.clip(gradient, -10.0, 10.0)
                    pooled = np.mean(sequence[0], axis=0)
                    update = np.outer(pooled, gradient)
                    update = np.clip(update, -3.5, 3.5)
                    self.output_projection = np.asarray(self.output_projection, dtype=np.float64) - update
                    self.output_projection = np.clip(self.output_projection, -50.0, 50.0)
                
                if len(self.encoder_layers) > 0:
                    pooled = np.mean(sequence[0], axis=0)
                    loss_gradient = np.asarray((probs - target) * learning_rate * 0.25, dtype=np.float64)
                    loss_gradient = np.clip(loss_gradient, -0.6, 0.6)
                    
                    for layer_idx in range(len(self.encoder_layers)):
                        layer = self.encoder_layers[layer_idx]
                        if 'W_o' in layer and pooled.shape[0] == layer['W_o'].shape[0]:
                            pooled_expanded = pooled.reshape(-1, 1)
                            gradient_expanded = loss_gradient.reshape(1, -1)
                            
                            if gradient_expanded.shape[1] <= layer['W_o'].shape[1]:
                                gradient_padded = np.pad(gradient_expanded, ((0, 0), (0, layer['W_o'].shape[1] - gradient_expanded.shape[1])), 'constant')
                            else:
                                gradient_padded = gradient_expanded[:, :layer['W_o'].shape[1]]
                            
                            layer_update = np.dot(pooled_expanded, gradient_padded)
                            layer_update = np.clip(layer_update, -0.4, 0.4)
                            layer['W_o'] = np.asarray(layer['W_o'], dtype=np.float64) - layer_update
                            layer['W_o'] = np.clip(layer['W_o'], -10.0, 10.0)
                        
                        if 'W_q' in layer and pooled.shape[0] == layer['W_q'].shape[0]:
                            pooled_expanded = pooled.reshape(-1, 1)
                            gradient_expanded = loss_gradient.reshape(1, -1)
                            
                            if gradient_expanded.shape[1] <= layer['W_q'].shape[1]:
                                gradient_padded = np.pad(gradient_expanded, ((0, 0), (0, layer['W_q'].shape[1] - gradient_expanded.shape[1])), 'constant')
                            else:
                                gradient_padded = gradient_expanded[:, :layer['W_q'].shape[1]]
                            
                            layer_update = np.dot(pooled_expanded, gradient_padded) * 0.03
                            layer_update = np.clip(layer_update, -0.2, 0.2)
                            layer['W_q'] = np.asarray(layer['W_q'], dtype=np.float64) - layer_update
                            layer['W_q'] = np.clip(layer['W_q'], -10.0, 10.0)
                        
                        if 'W_k' in layer and pooled.shape[0] == layer['W_k'].shape[0]:
                            pooled_expanded = pooled.reshape(-1, 1)
                            gradient_expanded = loss_gradient.reshape(1, -1)
                            
                            if gradient_expanded.shape[1] <= layer['W_k'].shape[1]:
                                gradient_padded = np.pad(gradient_expanded, ((0, 0), (0, layer['W_k'].shape[1] - gradient_expanded.shape[1])), 'constant')
                            else:
                                gradient_padded = gradient_expanded[:, :layer['W_k'].shape[1]]
                            
                            layer_update = np.dot(pooled_expanded, gradient_padded) * 0.03
                            layer_update = np.clip(layer_update, -0.2, 0.2)
                            layer['W_k'] = np.asarray(layer['W_k'], dtype=np.float64) - layer_update
                            layer['W_k'] = np.clip(layer['W_k'], -10.0, 10.0)
                        
                        if 'W_v' in layer and pooled.shape[0] == layer['W_v'].shape[0]:
                            pooled_expanded = pooled.reshape(-1, 1)
                            gradient_expanded = loss_gradient.reshape(1, -1)
                            
                            if gradient_expanded.shape[1] <= layer['W_v'].shape[1]:
                                gradient_padded = np.pad(gradient_expanded, ((0, 0), (0, layer['W_v'].shape[1] - gradient_expanded.shape[1])), 'constant')
                            else:
                                gradient_padded = gradient_expanded[:, :layer['W_v'].shape[1]]
                            
                            layer_update = np.dot(pooled_expanded, gradient_padded) * 0.03
                            layer_update = np.clip(layer_update, -0.2, 0.2)
                            layer['W_v'] = np.asarray(layer['W_v'], dtype=np.float64) - layer_update
                            layer['W_v'] = np.clip(layer['W_v'], -10.0, 10.0)
                        
                        if 'ff_w1' in layer and pooled.shape[0] == layer['ff_w1'].shape[0]:
                            pooled_expanded = pooled.reshape(-1, 1)
                            gradient_expanded = loss_gradient.reshape(1, -1)
                            
                            if gradient_expanded.shape[1] <= layer['ff_w1'].shape[1]:
                                gradient_padded = np.pad(gradient_expanded, ((0, 0), (0, layer['ff_w1'].shape[1] - gradient_expanded.shape[1])), 'constant')
                            else:
                                gradient_padded = gradient_expanded[:, :layer['ff_w1'].shape[1]]
                            
                            layer_update = np.dot(pooled_expanded, gradient_padded) * 0.015
                            layer_update = np.clip(layer_update, -0.15, 0.15)
                            layer['ff_w1'] = np.asarray(layer['ff_w1'], dtype=np.float64) - layer_update
                            layer['ff_w1'] = np.clip(layer['ff_w1'], -10.0, 10.0)
                        
                        if 'ff_w2' in layer:
                            pooled_expanded = pooled.reshape(-1, 1)
                            gradient_expanded = loss_gradient.reshape(1, -1)
                            
                            if pooled_expanded.shape[0] == layer['ff_w2'].shape[0]:
                                if gradient_expanded.shape[1] <= layer['ff_w2'].shape[1]:
                                    gradient_padded = np.pad(gradient_expanded, ((0, 0), (0, layer['ff_w2'].shape[1] - gradient_expanded.shape[1])), 'constant')
                                else:
                                    gradient_padded = gradient_expanded[:, :layer['ff_w2'].shape[1]]
                                
                                layer_update = np.dot(pooled_expanded, gradient_padded) * 0.015
                                layer_update = np.clip(layer_update, -0.15, 0.15)
                                layer['ff_w2'] = np.asarray(layer['ff_w2'], dtype=np.float64) - layer_update
                                layer['ff_w2'] = np.clip(layer['ff_w2'], -10.0, 10.0)
            
            accuracy = correct / len(X_train) if len(X_train) > 0 else 0
            avg_loss = total_loss / len(X_train) if len(X_train) > 0 else 0
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {accuracy:.4f}, Loss: {avg_loss:.4f}")
        
        if X_val is not None and y_val is not None:
            correct = 0
            history = []
            for i in range(len(X_val)):
                features = X_val[i]
                true_label = y_val[i]
                
                sequence, history = self.prepare_sequence(features, history)
                output, _ = self.forward(sequence)
                
                pred_class = np.argmax(output[0])
                
                if pred_class == true_label:
                    correct += 1
            
            val_accuracy = correct / len(X_val) if len(X_val) > 0 else 0
            print(f"Validation Accuracy: {val_accuracy:.4f}")
        
        return self
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'positional_encoding': self.positional_encoding,
                'encoder_layers': self.encoder_layers,
                'output_projection': self.output_projection,
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'n_layers': self.n_layers,
                'seq_length': self.seq_length,
                'feature_names': self.feature_names,
                'server_mapping': self.server_mapping
            }, f)
    
    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(
            d_model=data.get('d_model', 64),
            n_heads=data.get('n_heads', 4),
            n_layers=data.get('n_layers', 2),
            seq_length=data.get('seq_length', 10)
        )
        instance.positional_encoding = data.get('positional_encoding')
        instance.encoder_layers = data.get('encoder_layers', [])
        instance.output_projection = data.get('output_projection')
        instance.feature_names = data.get('feature_names')
        instance.server_mapping = data.get('server_mapping')
        return instance
