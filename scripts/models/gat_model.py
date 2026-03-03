
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATModel(nn.Module):
    def __init__(self, node_dim=74, edge_dim=4, hidden_dim=256, heads=4, output_dim=12):
        super().__init__()
        self.conv1 = GATConv(node_dim, hidden_dim, heads=heads, dropout=0.2)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # GAT layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        # Final layer
        x = self.fc(x)
        return x