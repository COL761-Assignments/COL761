import argparse
import pandas as pd
import torch
from torch_geometric.nn import GATConv, GATv2Conv, global_add_pool, Set2Set, global_max_pool, AttentionalAggregation, global_mean_pool
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR
import os


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


def make_weights_for_balanced_classes(labels, nclasses):                        
    count = [0] * nclasses
    for item in labels:                                                         
        int_item = int(item)  # Convert label to integer
        count[int_item] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N / float(count[i])                                 
    weight = [0] * len(labels)                                              
    for idx, val in enumerate(labels):                                          
        int_val = int(val)  # Convert label to integer
        weight[idx] = weight_per_class[int_val]                                     
    return weight



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


def train_and_validate(model, train_loader, val_loader, optimizer, num_epochs, task_type, scheduler=None):
    criterion = torch.nn.BCELoss() if task_type == 'classification' else torch.nn.MSELoss()
    train_losses, val_losses = [], []
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1) 

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = 0
            if task_type == 'classification':
                loss = criterion(out.view(-1), data.y.float())
            else:
                out = out.squeeze()
                loss = criterion(out, data.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))

        # Step the scheduler after each epoch
        if scheduler:
            scheduler.step()

        print(f'Epoch {epoch}, Training Loss: {train_losses[-1]}')

    return model, train_losses, val_losses




def main():
    parser = argparse.ArgumentParser(description="Training a classification model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    args = parser.parse_args()
    print(f"Training a classification model. Output will be saved at {args.model_path}. Dataset will be loaded from {args.dataset_path}. Validation dataset will be loaded from {args.val_dataset_path}.")
    Input_feature_dim = 9
    edge_attr_dim = 3
    num_classes = 1
    task_type = 'classification'
    train_loader = 0
    lr = 0
    hidden_channels = 0
    outchanels = 0
    num_epochs = 0

    val_loader = 0
    if task_type == 'classification':
        train_dataset = preprocess_and_create_dataset(args.dataset_path, nan = 1)
        # val_dataset = preprocess_and_create_dataset(args.val_dataset_path, nan = 0)
        labels = [data.y.item() for data in train_dataset]  # Extract labels from dataset
        weights = make_weights_for_balanced_classes(labels, 2)  
        weights = torch.DoubleTensor(weights)                                       
        sampler = WeightedRandomSampler(weights, len(weights))                     
        train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
        lr = 0.003
        hidden_channels =64
        outchanels = 32
        num_epochs = 6
        

    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


    model = GATNet(in_channels=Input_feature_dim, hidden_channels=hidden_channels, out_channels= outchanels , edge_attr_dim = edge_attr_dim, num_heads = 4, num_layers=3, task_type=task_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train and validate the model
    trained_model, train_losses, val_losses = train_and_validate(model, train_loader, val_loader, optimizer, num_epochs=num_epochs, task_type=task_type)
    
    torch.save(trained_model.state_dict(), args.model_path)



if __name__=="__main__":
    main()
    




    