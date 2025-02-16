### TransformersWithGCNN
This repository implements a hybrid model combining Graph Convolutional Neural Networks (GCNN) and Transformer-based layers using PyTorch Geometric. The model is applied to the Cora citation network dataset, leveraging Node2Vec embeddings and Principal Component Analysis (PCA) to improve node classification performance.

### Project Overview
The BalancedGCNTransformer model integrates:

-GCN Layer: Captures local neighborhood information.
-Transformer Layers: Exploits long-range dependencies within the graph.
-Node2Vec Embeddings: Provides additional semantic context.
-PCA for Dimensionality Reduction: Reduces high-dimensional feature space to improve model efficiency.
-Regularization Techniques: LayerNorm and Dropout to prevent overfitting.

The model is evaluated on the Cora dataset, a well-known citation network benchmark.

## Model Architecture

The BalancedGCNTransformer consists of:

-GCN Layer: Extracts localized structural patterns.
-Two TransformerConv Layers: Models global relationships across nodes.
-MLP Classifier: A two-layer MLP for final node classification.

## Forward Pass Steps:

GCN → TransformerConv (x2) → LayerNorm → Dropout → MLP → LogSoftmax


## Dataset
The model is trained and evaluated on the Cora citation network from PyTorch Geometric:

-Nodes: 2,708
-Edges: 5,429
-Features per Node: 1,433
-Classes: 7
-Preprocessing: PCA reduces feature dimensions to 256, followed by Node2Vec embeddings of size 128.


## Model Insights
The BalancedGCNTransformer is a hybrid architecture that combines GCN's localized learning with Transformer-based global attention. The addition of Node2Vec embeddings further enriches the node representations.

## Applications:

Citation Networks (like Cora)
Social Network Analysis
Recommendation Systems
