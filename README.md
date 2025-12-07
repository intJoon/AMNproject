# Machine Learning-Based Load Balancing in Software-Defined Networks

A comprehensive implementation and comparison of static vs. intelligent load balancing algorithms in SDN networks, featuring 10 different algorithms including traditional Round-Robin and 9 state-of-the-art machine learning approaches.

## Project Overview

This project implements and compares traditional Round-Robin load balancing with Machine Learning-based intelligent load balancing algorithms in a Software-Defined Network (SDN). The implementation demonstrates measurable improvements in routing accuracy, response time, and resource utilization.

**Team Members:**
- Joon
- Davin
- Mon

## Algorithms Implemented

### Baseline Algorithm
1. **Round Robin** - Static load balancing algorithm that distributes requests sequentially without considering server load or performance.

### Machine Learning Algorithms

2. **Random Forest** - Ensemble learning method using multiple decision trees for routing decisions. Achieves highest accuracy (~94-95%).
3. **Graph Coloring** - Graph-based algorithm using conflict graphs to model server relationships and optimal resource allocation.
4. **DDPG (Deep Deterministic Policy Gradient)** - Reinforcement learning algorithm with Q-learning and continuous action spaces. Shows significant improvement (~70-71% accuracy).
5. **PPO (Proximal Policy Optimization)** - Policy gradient method with clipped surrogate objectives, GAE, and trust-based policy updates using KL divergence.
6. **Improved DDPG** - Enhanced DDPG with SumTree-based prioritized experience replay for faster convergence through focused learning on critical transitions.
7. **Multi-Agent DDPG** - Multi-agent reinforcement learning with cooperation matrix for inter-agent coordination and distributed decision-making.
8. **Transformer** - Multi-head self-attention mechanism for temporal pattern recognition, inspired by Temporal Fusion Transformer (TFT).
9. **GAT (Graph Attention Network)** - Graph neural network with attention mechanisms for server relationship modeling and adaptive feature aggregation.
10. **TGNN (Temporal Graph Neural Network)** - Spatial-temporal graph convolution for dynamic network state prediction and proactive load balancing.

## Project Structure

```
Project/
├── data_processing/
│   └── preprocess_data.py          # Data preprocessing pipeline
├── dataset/
│   ├── training_data_250samples_*.json
│   └── training_data_300samples_*.json
├── models/
│   ├── random_forest_model.py       # Random Forest implementation
│   ├── graph_coloring_model.py      # Graph Coloring implementation
│   ├── ddpg_model.py                # DDPG implementation
│   ├── ppo_model.py                 # PPO implementation
│   ├── improved_ddpg_model.py       # Improved DDPG with SumTree
│   ├── multi_agent_model.py        # Multi-Agent DDPG
│   ├── transformer_model.py        # Transformer-based model
│   ├── gat_model.py                 # Graph Attention Network
│   └── tgnn_model.py                # Temporal Graph Neural Network
├── training/
│   └── train_models.py              # Training script for all models
├── evaluation/
│   ├── compare_algorithms.py        # Algorithm comparison and evaluation
│   ├── round_robin.py               # Round Robin baseline
│   └── visualizations/              # Generated visualization files
├── mininet_integration/
│   ├── load_balancer.py             # Mininet integration module
│   └── ryu_controller_example.py     # Ryu SDN controller example
├── saved_models/
│   ├── preprocessed_data.pkl
│   ├── random_forest.pkl
│   ├── graph_coloring.pkl
│   ├── ddpg.pkl
│   ├── ppo.pkl
│   ├── improved_ddpg.pkl
│   ├── multi_agent.pkl
│   ├── transformer.pkl
│   ├── gat.pkl
│   ├── tgnn.pkl
│   └── evaluation_results.pkl
└── requirements.txt
```

## Installation

### Prerequisites

- Python 3.8+
- Mininet 2.3+ (for network emulation)
- Ryu SDN Controller (for OpenFlow control)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Project
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

Additional dependencies may be required:
```bash
pip install torch torchvision  # For deep learning models
pip install reportlab          # For PDF generation
```

3. Install Mininet (if not already installed):
```bash
# Ubuntu/Debian
sudo apt-get install mininet

# Or build from source
git clone https://github.com/mininet/mininet
cd mininet
sudo ./util/install.sh -a
```

## Usage

### Complete Pipeline

Run the complete pipeline from data preprocessing to documentation generation:

```bash
# 1. Preprocess data
python data_processing/preprocess_data.py

# 2. Train all models
python training/train_models.py

# 3. Evaluate and compare algorithms
python evaluation/compare_algorithms.py

# 4. Generate documentation
python documentation/create_documentation.py
```

### Step-by-Step

#### 1. Data Preprocessing

Preprocess the training data and generate features:

```bash
python data_processing/preprocess_data.py
```

This will:
- Load JSON datasets from the `dataset/` folder
- Extract features (24 features including server statistics, response times, temporal patterns)
- Generate optimal server labels based on response time and success rate
- Split data into train/validation/test sets (64%/16%/20%)
- Save preprocessed data to `saved_models/preprocessed_data.pkl`

#### 2. Model Training

Train all models:

```bash
python training/train_models.py
```

This will train all 9 ML models:
- Random Forest (100 estimators, max depth 10)
- Graph Coloring
- DDPG (50 epochs)
- PPO (50 epochs)
- Improved DDPG (50 epochs)
- Multi-Agent DDPG (50 epochs)
- Transformer (30 epochs)
- GAT (20 epochs)
- TGNN (10 epochs)

All models are saved as `.pkl` files in `saved_models/`.

#### 3. Algorithm Evaluation

Compare all algorithms:

```bash
python evaluation/compare_algorithms.py
```

This will:
- Evaluate all 10 algorithms (Round Robin + 9 ML models)
- Generate performance metrics (accuracy, response time, improvement over baseline)
- Create visualization files in `evaluation/visualizations/`:
  - `performance_comparison.png`
  - `detailed_analysis.png`
  - `advanced_analysis.png`
  - `statistical_analysis.png`
  - `confusion_matrices.png`
- Save evaluation results to `saved_models/evaluation_results.pkl`

#### 4. Documentation Generation

Generate PDF documentation and summary images:

```bash
python documentation/create_documentation.py
```

This will:
- Load evaluation results
- Generate `summary_image.png` with comprehensive algorithm comparison
- Create `ML_Load_Balancing_Documentation.pdf` with detailed analysis

### Mininet Integration

Use the trained models in Mininet:

```python
from mininet_integration.load_balancer import get_load_balancer

# Load a trained model
lb = get_load_balancer(algorithm='random_forest')

# Select server for a request
server = lb.select_server()

# Update server statistics
lb.update_server_stats(server, response_time_ms=150, success=True)
```

Available algorithms:
- `'random_forest'`
- `'graph_coloring'`
- `'ddpg'`
- `'ppo'`
- `'improved_ddpg'`
- `'multi_agent'`
- `'transformer'`
- `'gat'`
- `'tgnn'`
- `'round_robin'`

## Algorithm Comparison

The project evaluates algorithms based on:

- **Accuracy**: Percentage of correct server selections
- **Response Time**: Average response time in milliseconds
- **Improvement**: Performance improvement over Round Robin baseline

### Key Results

- **Random Forest** typically achieves the highest accuracy (~94-95%) with +215% improvement over baseline
- **DDPG** shows significant improvement over baseline (~70-71% accuracy, +136% improvement)
- **Advanced algorithms** (PPO, Transformer, GAT, TGNN) represent 2025 state-of-the-art approaches
- All ML algorithms consistently outperform static Round Robin
- Feature engineering plays a crucial role in model performance

Detailed performance metrics and visualizations are available in:
- `documentation/ML_Load_Balancing_Documentation.pdf`
- `evaluation/visualizations/`

## Experimental Setup

- **Network Emulator**: Mininet 2.3+
- **SDN Controller**: Ryu (Python-based OpenFlow controller)
- **Topology**: 12 host nodes, 4 server nodes (h1, h2, h3, h4), 1 OpenFlow switch
- **Dataset**: 550 samples collected from real network experiments
- **Features**: 24 features including:
  - Server statistics (mean, std, min, max response times)
  - Rolling averages and recent performance metrics
  - Temporal features (hour, minute, second)
  - Server-specific recent performance indicators
- **Data Split**: 352/88/110 samples (64%/16%/20%) for train/validation/test
- **Servers**: h1 (fastest), h2 (slowest), h3 (second fastest), h4 (moderate)

## Model Differences

### Random Forest
- **Type**: Ensemble ML
- **Approach**: Multiple decision trees voting
- **Best for**: High accuracy, interpretable results
- **Configuration**: 100 estimators, max depth 10
- **Performance**: ~94-95% accuracy

### DDPG
- **Type**: Reinforcement Learning
- **Approach**: Q-learning with continuous actions
- **Best for**: Learning optimal policies through exploration
- **Training**: 50 epochs
- **Performance**: ~70-71% accuracy

### PPO
- **Type**: Reinforcement Learning
- **Approach**: Trust-based policy updates with KL divergence
- **Best for**: Stable policy learning
- **Features**: Clipped surrogate objective, GAE
- **Reference**: MDPI Applied Sciences 2023, ArXiv 2025

### Improved DDPG
- **Type**: Reinforcement Learning
- **Approach**: Prioritized experience replay with SumTree
- **Best for**: Faster convergence through focused learning
- **Features**: TD-error based sampling, importance sampling weights
- **Reference**: Space Frontiers 2022

### Multi-Agent DDPG
- **Type**: Multi-Agent RL
- **Approach**: Distributed decision-making with cooperation matrix
- **Best for**: Collaborative server coordination
- **Features**: 4 server agents, dynamic cooperation, heterogeneous GNN integration
- **Reference**: Springer 2025

### Transformer
- **Type**: Deep Learning
- **Approach**: Multi-head self-attention for temporal patterns
- **Best for**: Capturing long-term dependencies
- **Features**: Positional encoding, encoder layers, sequence processing
- **Reference**: ArXiv 2025, Nature Scientific Reports 2025

### GAT
- **Type**: Graph Neural Network
- **Approach**: Attention-based graph convolution
- **Best for**: Modeling server relationships
- **Features**: Multi-head attention, bi-level attention, neighbor importance weighting
- **Reference**: ArXiv 2023-2024

### TGNN
- **Type**: Temporal Graph Neural Network
- **Approach**: Spatial-temporal graph convolution
- **Best for**: Dynamic network state prediction
- **Features**: Temporal window, dynamic adjacency based on performance similarity
- **Reference**: ScienceDirect 2023, PubMed 2024

### Graph Coloring
- **Type**: Graph-based
- **Approach**: Conflict graph modeling
- **Best for**: Optimal resource allocation
- **Features**: Weighted conflict matrices, server relationship modeling

## Workflow

1. **Data Collection**: Network experiments generate JSON datasets with server performance metrics
2. **Preprocessing**: Feature extraction and label generation from raw data
3. **Training**: Multiple ML models trained on preprocessed data
4. **Evaluation**: Comprehensive comparison of all algorithms
5. **Visualization**: Generation of performance charts and analysis
6. **Documentation**: PDF report with detailed results and methodology
7. **Deployment**: Integration with Mininet/Ryu for real-time load balancing

## References

- Machine Learning Load Balancing Algorithms in SDN-enabled Massive IoT Networks (IEEE IPCCC 2023)
- Trust-Based Proximal Policy Optimization for SDN Load Balancing (MDPI Applied Sciences 2023)
- Deep Reinforcement Learning for QoS-Aware Load Balancing in 5G Networks (ArXiv 2025)
- Graph Reinforcement Learning in Open Radio Access Networks (ArXiv 2025)
- Automatic Load-Balancing Architecture Based on Reinforcement Learning (Space Frontiers 2022)
- Spatial-Temporal Graph Neural Networks for Network Traffic Prediction (ScienceDirect 2023)
- Temporal Graph Neural Networks for Dynamic Resource Management (PubMed 2024)
- Transformer-based Time Series Forecasting for Network Load Prediction (ArXiv 2025, Nature 2025)

## License

This project is developed for academic research purposes.

## Contact

For questions or issues, please contact the project team members.
