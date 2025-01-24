import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
from dgl.nn import GraphConv, SAGEConv, GATConv, GINConv

class Hadamard_MLPPredictor(nn.Module):
    def __init__(self, h_feats, dropout, layer=2, res=False):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(h_feats, h_feats))
        for _ in range(layer - 2):
            self.lins.append(torch.nn.Linear(h_feats, h_feats))
        self.lins.append(torch.nn.Linear(h_feats, 1))
        self.dropout = dropout
        self.res = res

    def forward(self, x_i, x_j):
        x = x_i * x_j
        ori = x
        for lin in self.lins[:-1]:
            x = lin(x)
            if self.res:
                x = x + ori
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x.squeeze()

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, norm=False, dp4norm=0.2, drop_edge=False, relu=False, prop_step=2, residual=0, conv='GCN'):
        super(GCN, self).__init__()
        if conv == 'GCN':
            self.conv1 = GraphConv(in_feats, h_feats)
            self.conv2 = GraphConv(h_feats, h_feats)
        elif conv == 'SAGE':
            self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
            self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
        elif conv == 'GAT':
            self.conv1 = GATConv(in_feats, h_feats // 4, 4)
            self.conv2 = GATConv(h_feats, h_feats // 4, 4)
        elif conv == 'GIN':
            self.conv1 = GINConv(nn.Linear(in_feats, h_feats), 'mean')
            self.conv2 = GINConv(nn.Linear(h_feats, h_feats), 'mean')
        self.norm = norm
        self.drop_edge = drop_edge
        self.relu = relu
        self.prop_step = prop_step
        self.residual = residual
        if norm:
            self.ln = nn.LayerNorm(h_feats)
            self.dp = nn.Dropout(dp4norm)

    def forward(self, g, in_feat):
        ori = in_feat
        if self.drop_edge:
            g = drop_edge(g)
        h = self.conv1(g, in_feat).flatten(1) + self.residual * ori
        for i in range(1, self.prop_step):
            if self.relu:
                h = F.relu(h)
            if self.norm:
                h = self.ln(h)
                h = self.dp(h)
            h = self.conv2(g, h).flatten(1) + self.residual * ori
        return h

def drop_edge(g, dpe=0.2):
    """Randomly drop a percentage of edges from the graph."""
    g = g.clone()
    num_edges = g.number_of_edges()
    edge_ids = torch.randperm(num_edges)[:int(num_edges * dpe)].to(g.device)
    g.remove_edges(edge_ids)
    g = dgl.add_self_loop(g)
    return g

# Load the trade dataset
df = pd.read_csv(r'D:\Final Year Project\Project\GAE\machinery_2018.csv')

# Filter the dataset for a specific year (e.g., 2018)
df = df[df['year'] == 2018]

# Create a mapping of country IDs to graph node IDs
unique_countries = pd.concat([df['exporter_id'], df['importer_id']]).unique()
country_to_id = {country: idx for idx, country in enumerate(unique_countries)}

# Map countries to node IDs
df['exporter_node'] = df['exporter_id'].map(country_to_id)
df['importer_node'] = df['importer_id'].map(country_to_id)

# Create a graph
g = dgl.graph((torch.tensor(df['exporter_node'].values, dtype=torch.int64), 
               torch.tensor(df['importer_node'].values, dtype=torch.int64)))

# Add edge features (normalized trade value for each HS code)
df['hs_code_value'] = df.groupby('hs_code')['value'].transform('sum')
df['normalized_hs_value'] = MinMaxScaler().fit_transform(df[['hs_code_value']])

# Combine HS code and normalized value as edge features
edge_features = df[['hs_code', 'normalized_hs_value']].values
g.edata['features'] = torch.tensor(edge_features, dtype=torch.float32)

# Add node features (initialize with dummy features, e.g., degrees)
g.ndata['feat'] = torch.ones((g.num_nodes(), 1), dtype=torch.float32)

# Split the dataset into train and test edges
u, v = g.edges()
labels = g.edata['features'][:, 1]  # Use normalized trade value as labels

# Split edges into train and test sets
u_train, u_test, v_train, v_test, labels_train, labels_test = train_test_split(
    u.numpy(), v.numpy(), labels.numpy(), test_size=0.2, random_state=42
)

# Create train and test graphs
g_train = dgl.edge_subgraph(g, torch.tensor(np.arange(len(u_train), dtype=np.int64)))
g_test = dgl.edge_subgraph(g, torch.tensor(np.arange(len(u_test), dtype=np.int64)))

# Extract node features
node_features = g.ndata['feat']

# Initialize the model, optimizer, and loss function
in_feats = node_features.shape[1]
h_feats = 128  # Hidden layer size
model = GCN(in_feats, h_feats, conv='GCN')  # Use other convolution types as desired
predictor = Hadamard_MLPPredictor(h_feats, dropout=0.2, layer=3, res=True)

optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=0.01)
loss_fn = torch.nn.MSELoss()

# Training loop
for epoch in range(50):
    model.train()
    g_train = drop_edge(g_train, dpe=0.2)  # Optionally drop edges for training
    h = model(g_train, node_features)

    # Get source and target node embeddings
    h_src = h[g_train.edges()[0]]
    h_dst = h[g_train.edges()[1]]

    # Compute predictions and loss
    logits = predictor(h_src, h_dst)
    loss = loss_fn(logits, g_train.edata['features'][:, 1])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{50}, Loss: {loss.item():.4f}")

# Evaluation
def evaluate(g_test, node_features, model, predictor):
    model.eval()
    with torch.no_grad():
        h = model(g_test, node_features)
        h_src = h[g_test.edges()[0]]
        h_dst = h[g_test.edges()[1]]
        logits = predictor(h_src, h_dst)
        preds = logits  # For regression, predictions are raw logits
        labels = g_test.edata['features'][:, 1]
        mse = torch.nn.functional.mse_loss(preds, labels).item()
        return mse

mse = evaluate(g_test, node_features, model, predictor)
print(f"Test MSE: {mse:.4f}")

def generate_negative_edges(g, num_samples):
    """
    Generate a specified number of negative edges (non-existent edges) for a graph.
    Args:
        g: The graph from which to generate negative edges.
        num_samples: Number of negative edges to generate.
    Returns:
        A tensor of shape (2, num_samples), representing the source and destination nodes of negative edges.
    """
    num_nodes = g.num_nodes()
    existing_edges = set(zip(g.edges()[0].tolist(), g.edges()[1].tolist()))
    negative_edges = []
    
    while len(negative_edges) < num_samples:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        if u != v and (u, v) not in existing_edges and (v, u) not in existing_edges:
            negative_edges.append((u, v))
            existing_edges.add((u, v))
    
    return torch.tensor(negative_edges, dtype=torch.long).t()


# Predict top potential new links with commodity (HS code)
all_possible_edges = generate_negative_edges(g, 1000)  # Generate 1000 random potential links
model.eval()
with torch.no_grad():
    h = model(g, node_features)
    h_src = h[all_possible_edges[0]]
    h_dst = h[all_possible_edges[1]]
    predictions = predictor(h_src, h_dst)

# Sort potential links by predicted score
sorted_predictions = predictions.argsort(descending=True)

top_links = all_possible_edges[:, sorted_predictions[:1000]]


def get_country_by_id(country_id, country_to_id):
    id_to_country = {v: k for k, v in country_to_id.items()}
    return id_to_country.get(country_id, "Unknown Country")

# Map edges to HS codes and predictions
top_links_with_commodities = []
for idx in sorted_predictions[:100]:  # Top 10 links
    src, dst = all_possible_edges[:, idx]
    hs_code = g.edata['features'][idx, 0].item()  # Retrieve HS code from edge features
    top_links_with_commodities.append((src.item(), dst.item(), hs_code, predictions[idx].item()))

print("Top 10 Predicted Links with Commodities (Source, Target, HS Code, Score):")
for src, dst, hs_code, score in top_links_with_commodities:
    src_name = get_country_by_id(src, country_to_id)
    dst_name = get_country_by_id(dst, country_to_id)
    print(f"{src_name} -> {dst_name}, HS Code: {hs_code}, Predicted Score: {score:.4f}")



'''----------------------------------------------------------------------------------------------------'''
df_2019 = pd.read_csv('.\Dataset\machinery_2019.csv')



df_2019['exporter_node'] = df_2019['exporter_id'].map(country_to_id)
df_2019['importer_node'] = df_2019['importer_id'].map(country_to_id)
g_2019 = dgl.graph((torch.tensor(df_2019['exporter_node'].values, dtype=torch.int64), 
                    torch.tensor(df_2019['importer_node'].values, dtype=torch.int64)))
df_2019['hs_code_value'] = df_2019.groupby('hs_code')['value'].transform('sum')
df_2019['normalized_hs_value'] = MinMaxScaler().fit_transform(df_2019[['hs_code_value']])
g_2019.edata['features'] = torch.tensor(df_2019[['hs_code', 'normalized_hs_value']].values, dtype=torch.float32)


# Convert edges in 2019 graph to a set for quick lookup
edges_2019 = set(zip(df_2019['exporter_node'], df_2019['importer_node']))

matched_links = []
for src, dst, hs_code, score in top_links_with_commodities:
    if (src, dst) in edges_2019:
        matched_links.append((src, dst, hs_code, score))

print(f"Number of predicted links present in 2019 data: {len(matched_links)}")
print("Matched Links with HS Code and Score:")
for src, dst, hs_code, score in matched_links:
    src_name = get_country_by_id(src, country_to_id)
    dst_name = get_country_by_id(dst, country_to_id)
    print(f"{src_name} -> {dst_name}, HS Code: {hs_code}, Score: {score:.4f}")


for src, dst, hs_code, score in matched_links:
    actual_value = df_2019[(df_2019['exporter_node'] == src) & (df_2019['importer_node'] == dst) & 
                           (df_2019['hs_code'] == hs_code)]['value'].sum()
    print(f"Predicted: {score:.4f}, Actual Trade Value in 2019: {actual_value:.2f}")



'''----------------------------------------------------------------------------------------------------'''

from sklearn.metrics import roc_auc_score

def calculate_auc(g_test, model, predictor, node_features, num_negative_samples=1000):
    # Positive edges from the test graph
    pos_u, pos_v = g_test.edges()
    pos_labels = torch.ones(pos_u.shape[0])

    # Generate negative edges
    negative_edges = generate_negative_edges(g_test, num_negative_samples)
    neg_u, neg_v = negative_edges
    neg_labels = torch.zeros(neg_u.shape[0])

    # Combine positive and negative edges
    all_u = torch.cat([pos_u, neg_u])
    all_v = torch.cat([pos_v, neg_v])
    all_labels = torch.cat([pos_labels, neg_labels])

    # Compute node embeddings
    model.eval()
    with torch.no_grad():
        h = model(g_test, node_features)
        h_src = h[all_u]
        h_dst = h[all_v]
        all_scores = predictor(h_src, h_dst)

    # Calculate AUC
    auc = roc_auc_score(all_labels.numpy(), all_scores.numpy())
    return auc

# Incorporate into the pipeline

auc = calculate_auc(g_test, model, predictor, node_features)
print(f"Test AUC: {auc:.4f}")

