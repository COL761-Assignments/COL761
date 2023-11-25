import pandas as pd
from sklearn.linear_model import LogisticRegression
import os
import numpy as np

def preprocess_and_create_dataset(path):
    graph_labels = pd.read_csv(os.path.join(path, "graph_labels.csv.gz"), header=None).squeeze().tolist()
    num_nodes = pd.read_csv(os.path.join(path, "num_nodes.csv.gz"), header=None).squeeze().tolist()
    num_edges = pd.read_csv(os.path.join(path, "num_edges.csv.gz"), header=None).squeeze().tolist()

    # Load node_features, edges, and edge_features
    node_features = pd.read_csv(os.path.join(path, "node_features.csv.gz"), header=None)
    edges = pd.read_csv(os.path.join(path, "edges.csv.gz"), header=None)
    edge_features = pd.read_csv(os.path.join(path, "edge_features.csv.gz"), header=None)

    # Initialize data list and counters for nodes and edges
    data_list = []
    node_counter = 0
    edge_counter = 0
    y = []

    for i, label in enumerate(graph_labels):
        if pd.isna(label):              
            # Skip graphs with NaN labels
            node_counter += num_nodes[i]
            edge_counter += num_edges[i]
            continue

        # Extracting and encoding node and edge data
        nodes_end = node_counter + num_nodes[i]
        edges_end = edge_counter + num_edges[i]

        node_data = node_features.iloc[node_counter:nodes_end].values.mean(axis=0)
        edge_attr_data = edge_features.iloc[edge_counter:edges_end].values.mean(axis=0)

        # Convert label based on the task type
        label = int(label) 
        y.append(label)

        data = np.concatenate((node_data, edge_attr_data), axis=None)
        data_list.append(data)

        # Update counters
        node_counter = nodes_end
        edge_counter = edges_end

    return np.array(data_list),np.array(y)

# Load training data
path = '/home/vignesh/Desktop/SEM_7_2023-24/COL761/A3/dataset/dataset_2/train/'

X_train,y_train = preprocess_and_create_dataset(path)

print(X_train.shape)
print(y_train.shape)


# Load validation data
path = '/home/vignesh/Desktop/SEM_7_2023-24/COL761/A3/dataset/dataset_2/valid/'

X_val,y_val = preprocess_and_create_dataset(path)



# Initialize and train the logistic regression model on the training set
model = LogisticRegression()
model.fit(X_train, y_train)

from sklearn.metrics import roc_auc_score

predictions_valid_proba = model.predict_proba(X_val)[:, 1]  # Assuming binary classification

# Calculate ROC AUC score
roc_score = roc_auc_score(y_val, predictions_valid_proba)

print("ROC AUC Score:", roc_score)
