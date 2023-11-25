import argparse
import pandas as pd
import torch
from torch_geometric.nn import GATConv, GATv2Conv, global_add_pool, Set2Set, global_max_pool, AttentionalAggregation, global_mean_pool
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.nn import LayerNorm
import os
import  numpy as np
from help import Evaluator
from torch.optim.lr_scheduler import StepLR  # Example scheduler
import csv


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, edge_attr_dim, num_heads, num_layers, task_type, dropout_rate=0.0, use_v2=True ):
        super(GATNet, self).__init__()
        self.num_layers = num_layers
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()

        Conv = GATv2Conv if use_v2 else GATConv

        for i in range(num_layers):
            concat = True if i < num_layers - 1 else False
            heads = num_heads if i < num_layers - 1 else 1
            layer_out_channels = hidden_channels if i < num_layers - 1 else out_channels

            if concat and layer_out_channels % heads != 0:
                raise ValueError(f"'out_channels' must be divisible by 'heads' when 'concat' is True")

            conv_out_channels = layer_out_channels // heads if concat else layer_out_channels
            self.conv_layers.append(Conv(in_channels, conv_out_channels, heads=heads, concat=concat, dropout=dropout_rate, edge_dim=edge_attr_dim, bias=True))
            self.norm_layers.append(torch.nn.BatchNorm1d(layer_out_channels))
            
            in_channels = layer_out_channels

        # # Global Attention Pooling
        gate_nn = torch.nn.Linear(out_channels, 1)
        self.attentional_aggregation = AttentionalAggregation(gate_nn)

        # # Set2Set pooling
        self.set2set = Set2Set(in_channels, processing_steps=3)

        # Adjust the pooled features dimension for the MLP
        pooled_features_dim = 3 * in_channels + out_channels * 3  # Set2Set returns 2 * in_channels, others return out_channels

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(pooled_features_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        self.task_type = task_type

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for i in range(self.num_layers):
            x = self.dropout(x)
            x = self.conv_layers[i](x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)  # Removed skip connection
            x = self.norm_layers[i](x)

        x_sum = global_add_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_mean = global_mean_pool(x, batch)
        x_att = self.attentional_aggregation(x, batch)

        # Apply Set2Set pooling
        x_set2set = self.set2set(x, batch)

        # Concatenate all pooling outputs
        x = torch.cat([x_sum ,  x_max, x_mean, x_att, x_set2set], dim=1)

        if self.task_type == 'classification':
            x = torch.sigmoid(self.mlp(x))
        else:
            x = self.mlp(x).squeeze(-1)

        return x


class CustomGraphDataset(Dataset):
    def __init__(self, data_list):
        super(CustomGraphDataset, self).__init__()
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

def preprocess_and_create_dataset(path, nan=1, task_type='classification'):
    # Load num_nodes and num_edges
    num_nodes = pd.read_csv(os.path.join(path, "num_nodes.csv.gz"), header=None).squeeze().tolist()
    num_edges = pd.read_csv(os.path.join(path, "num_edges.csv.gz"), header=None).squeeze().tolist()

    # Load node_features, edges, and edge_features
    node_features = pd.read_csv(os.path.join(path, "node_features.csv.gz"), header=None)
    edges = pd.read_csv(os.path.join(path, "edges.csv.gz"), header=None)
    edge_features = pd.read_csv(os.path.join(path, "edge_features.csv.gz"), header=None)

    # Load graph_labels if nan is not 0
    if nan:
        graph_labels = pd.read_csv(os.path.join(path, "graph_labels.csv.gz"), header=None).squeeze().tolist()
    else:
        graph_labels = [None] * len(num_nodes)  # Placeholder labels

    # Initialize data list and counters for nodes and edges
    data_list = []
    node_counter = 0
    edge_counter = 0

    for i, label in enumerate(graph_labels):
        if nan :
            if pd.isna(label):
                node_counter += num_nodes[i]
                edge_counter += num_edges[i]
                continue

        # Extracting and encoding node and edge data
        nodes_end = node_counter + num_nodes[i]
        edges_end = edge_counter + num_edges[i]

        node_data = torch.tensor(node_features.iloc[node_counter:nodes_end].values, dtype=torch.float)
        edge_data = torch.tensor(edges.iloc[edge_counter:edges_end].values, dtype=torch.long).t().contiguous()
        edge_attr_data = torch.tensor(edge_features.iloc[edge_counter:edges_end].values, dtype=torch.float)

        if pd.isna(label):
            label = float('nan')

        # Convert label based on the task type
        label = int(label) if task_type == 'classification' and not pd.isna(label) else float(label)
        label_tensor_type = torch.long if task_type == 'classification' and not pd.isna(label) else torch.float
        data = Data(x=node_data, edge_index=edge_data, edge_attr=edge_attr_data, y=torch.tensor([label], dtype=label_tensor_type))

        data_list.append(data)

        # Update counters
        node_counter = nodes_end
        edge_counter = edges_end

    return CustomGraphDataset(data_list)



def evaluate_model(model, loader, task_type):
    model.eval()
    
    if task_type == 'classification':
        correct = 0
        total = 0
        predictions = []
        with torch.no_grad():
            for data in loader:
                out = model(data)
                predicted_label = (out > 0.5).float()  # Threshold for binary classification
                predictions.append(predicted_label.item())
                total += data.y.size(0)

        accuracy = correct / total
        return accuracy, predictions
    elif task_type == 'regression':
        total_loss = 0
        total_count = 0
        predictions = []

        with torch.no_grad():
            for data in loader:
                out = model(data)
                predictions.append(out.item())
                total_count += data.y.size(0)

        mse = total_loss / total_count
        rmse = mse ** 0.5
        return mse, rmse, predictions
    
def tocsv(y_arr, *, task):
    r"""Writes the numpy array to a csv file.
    params:
        y_arr: np.ndarray. A vector of all the predictions. Classes for
        classification and the regression value predicted for regression.

        task: str. Must be either of "classification" or "regression".
        Must be a keyword argument.
    Outputs a file named "y_classification.csv" or "y_regression.csv" in
    the directory it is called from. Must only be run once. In case outputs
    are generated from batches, only call this output on all the predictions
    from all the batches collected in a single numpy array. This means it'll
    only be called once.

    This code ensures this by checking if the file already exists, and does
    not over-write the csv files. It just raises an error.

    Finally, do not shuffle the test dataset as then matching the outputs
    will not work.
    """
    assert task in ["classification", "regression"], f"task must be either \"classification\" or \"regression\". Found: {task}"
    assert isinstance(y_arr, np.ndarray), f"y_arr must be a numpy array, found: {type(y_arr)}"
    assert len(y_arr.squeeze().shape) == 1, f"y_arr must be a vector. shape found: {y_arr.shape}"
    assert not os.path.isfile(f"y_{task}.csv"), f"File already exists. Ensure you are not calling this function multiple times (e.g. when looping over batches). Read the docstring. Found: y_{task}.csv"
    y_arr = y_arr.squeeze()
    df = pd.DataFrame(y_arr)
    df.to_csv(f"y_{task}.csv", index=False, header=False)

def test(model, test_loader, device):
    model.eval()
    all_ys = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch.x, batch.edge_index, batch.batch)
            ys_this_batch = output.cpu().numpy().tolist()
            all_ys.extend(ys_this_batch)
    numpy_ys = np.asarray(all_ys)
    tocsv(numpy_ys, task="classification") # <- Called outside the loop. Called in the eval code.


def main():
    parser = argparse.ArgumentParser(description="Evaluating the classification model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    args = parser.parse_args()
    print(f"Evaluating the classification model. Model will be loaded from {args.model_path}. Test dataset will be loaded from {args.dataset_path}.")
    val_accuracy, val_predictions = 0, 0

    task_type = 'classification'
    lr = 0.003
    in_channels = 9
    hidden_channels =64
    out_channels = 32
    model = GATNet(in_channels, hidden_channels, out_channels, edge_attr_dim=3, num_heads=4, num_layers=3 , task_type = task_type, dropout_rate=0.0, use_v2=True)
    model.load_state_dict(torch.load(args.model_path))

    val_dataset = preprocess_and_create_dataset(args.dataset_path, nan = 0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    val_accuracy, val_predictions = evaluate_model(model, val_loader, task_type)

    # print(val_predictions)

    # graph_labels = []

    # if task_type == 'classification':
    #     graph_labels = pd.read_csv(os.path.join(args.dataset_path, "graph_labels.csv.gz"), header=None).squeeze().tolist()


    # Convert to PyTorch tensors
    y_pred_numpy = np.array(val_predictions)
    tocsv(y_pred_numpy, task="classification")

    # y_true_tensor = torch.tensor(graph_labels).unsqueeze(1)
    # y_pred_tensor = torch.tensor(val_predictions).unsqueeze(1)

    # # Create the input dictionary for the evaluator
    # input_dict_tensor = {'y_true': y_true_tensor, 'y_pred': y_pred_tensor}

    # # Evaluate
    # if task_type == 'classification':
    #     evaluator = Evaluator('dataset2')
    #     result = evaluator.eval(input_dict_tensor)

    # print(result)


if __name__=="__main__":
    main()
    

