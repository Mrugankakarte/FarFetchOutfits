
from typing import Tuple, List

import numpy as np
from numpy.random import permutation
import matplotlib.pyplot as plt
import pandas as pd
import random
import torch
import torch.nn.functional as F

from torch import Tensor

from torch_geometric.nn import SAGEConv, to_hetero, GATv2Conv
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.sampler import NegativeSampling

from tqdm.auto import tqdm
import time

tqdm.pandas()

import networkx as nx
import plotly.graph_objects as go
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx
from collections import defaultdict

def bpr_loss(pos_scores, neg_scores):
    diff_score = pos_scores-neg_scores
    loss = -F.logsigmoid(diff_score).sum()
    return loss

class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=256, out_channels=128, heads=5, layer_type='GAT'):
        super().__init__()

        if layer_type == "SAGEConv":
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)

        if layer_type == "GAT":
            self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, add_self_loops=False, dropout=0.4)
            self.conv2 = GATv2Conv(hidden_channels*heads, out_channels, heads=1, concat=False, add_self_loops=False, dropout=0.4)

        self.relu = torch.nn.ReLU()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        return x

class Predictor(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.layer_class = torch.nn.Linear(out_channels, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.layer_class(x)

class Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, num_nodes_dict, graph_metadata, layer_type='GAT', device='cpu'):
        super().__init__()

        self.category_embd = torch.nn.Embedding(num_nodes_dict["category"], in_channels)

        pembds = pd.read_parquet("data/product_text_embedding.parquet")
        pembds = torch.from_numpy(pembds.drop("product_node_id", axis=1).values)
        self.product_emb = torch.nn.Embedding.from_pretrained(pembds, freeze=True)
        del pembds

        self.layer = torch.nn.Linear(1024, in_channels)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, heads=heads, layer_type=layer_type)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=graph_metadata, aggr="sum")

        self.pred_layer = Predictor(out_channels=out_channels)

        #self.l_func = torch.nn.TripletMarginLoss(margin=1.0, p=2.0)
        self.l_func = torch.nn.BCEWithLogitsLoss()
        self.alpha = 0.5
        self.device = device

    def inference(self, batch: HeteroData):

        x_dict = {
            "category": self.category_embd(batch['category'].node_id),
            "product": self.layer(self.product_emb(batch['product'].node_id))
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, batch.edge_index_dict)

        return x_dict


    def forward(self, batch: HeteroData, edge_types_to_predict: List) -> tuple[dict,Tensor]:

        x_dict = self.inference(batch)

        total_loss = 0.0
        for edge_type in edge_types_to_predict:
            #print(edge_type)
            # Get the source and destination node types from the edge type tuple
            src_node_type, _, dst_node_type = edge_type

            u = batch[src_node_type]['src_index']
            v = batch[dst_node_type]['dst_pos_index']
            vn = batch[dst_node_type]['dst_neg_index']

            u_feats = x_dict['product'][u]
            v_feats = x_dict['product'][v]
            vn_feats = x_dict['product'][vn]

            #total_loss += self.l_func(u_feats, v_feats, vn_feats)

            pscore = self.pred_layer(u_feats * v_feats)
            nscore = self.pred_layer(u_feats * vn_feats)

            # BCE loss
            '''
            pl = torch.ones_like(pscore)
            nl = torch.zeros_like(nscore)

            y_true = torch.cat((pl, nl))
            y_pred = torch.cat((pscore, nscore))

            total_loss += self.l_func(y_pred, y_true)
            '''

            #BPR
            total_loss += bpr_loss(pscore, nscore)

        return total_loss


def main():
    graph = torch.load("data/graph/graph.pt", weights_only=False)
    print(graph)

    num_node_dict = {}

    for node_type in graph.node_types:
        num_node_dict[node_type] = graph[node_type].num_nodes

    print(num_node_dict)
    training_edges = [('product', 'product_outfit', 'product')]

    transform = T.RandomLinkSplit(num_val=0.1, num_test=0.1,
                                  disjoint_train_ratio=0.1,
                                  is_undirected=True,
                                  add_negative_train_samples=False,
                                  edge_types=training_edges)
    train_data, val_data, test_data = transform(graph)

    train_prod_dataloader = LinkNeighborLoader(
        train_data,
        num_workers=3,
        num_neighbors=[32, 8],
        batch_size=64,
        neg_sampling=NegativeSampling(mode="triplet", amount=1),
        directed=False,
        shuffle=True,
        # edge_label=train_data[("product", "product_outfit", "product")].edge_label,
        edge_label_index=(('product', 'product_outfit', 'product'),
                          train_data[('product', 'product_outfit', 'product')].edge_label_index)
    )

    val_prod_dataloader = LinkNeighborLoader(
        val_data,
        num_workers=3,
        num_neighbors=[32, 8],
        batch_size=64,
        neg_sampling=NegativeSampling(mode="triplet", amount=1),
        directed=False,
        shuffle=True,
        # edge_label=train_data[("product", "product_outfit", "product")].edge_label,
        edge_label_index=(('product', 'product_outfit', 'product'),
                          val_data[('product', 'product_outfit', 'product')].edge_label_index)
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_channels = 64
    model = Model(in_channels=64, hidden_channels=64, out_channels=out_channels, heads=8,
                  num_nodes_dict=num_node_dict, graph_metadata=graph.metadata(),
                  layer_type='GAT', device=device)
    try:
        model.load_state_dict(torch.load("model_weights.pt", weights_only=True, map_location='cpu'))
        print("loaded model weights...")
    except:
        pass
    model.to(device)

    print(model)
    print(f"model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    train_epoch_loss = []
    val_epoch_loss = []
    scaler = torch.amp.GradScaler(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00001)

    EPOCHS = 5

    for epoch in range(EPOCHS):

        with tqdm(train_prod_dataloader, desc="Training") as tq:
            model.train()
            iter_loss = []
            for batch in tq:
                batch = batch.to(device)
                with torch.amp.autocast(device):
                    loss = model(batch, training_edges)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                # scaler.scale(loss).backward()
                # scaler.unscale_(opt)  # Unscale the gradients before clipping
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # scaler.step(opt)
                # scaler.update()
                opt.step()

                iter_loss.append(loss.detach().cpu().numpy())

                tq.set_postfix({'epoch': '%d' % epoch, 'loss': '%.03f' % loss}, refresh=False)

        with tqdm(val_prod_dataloader, desc="Validation") as tq, torch.no_grad(), torch.amp.autocast(device):
            model.eval()
            val_iter_loss = []
            for batch in tq:
                batch = batch.to(device)
                val_loss = model(batch, training_edges)

                val_iter_loss.append(val_loss.detach().cpu().numpy())

                tq.set_postfix({'epoch': '%d' % epoch, 'loss': '%.03f' % val_loss}, refresh=False)

        train_epoch_loss.append(np.mean(iter_loss))
        val_epoch_loss.append(np.mean(val_iter_loss))
        print(f"Epoch: {epoch} loss: {train_epoch_loss[epoch]} val loss:{val_epoch_loss[epoch]}")

    print("saving model_weights")
    torch.save(model.state_dict(), "model_weights.pt")

    print("generating node embeddings")
    product_nodeloader = NeighborLoader(graph,
                                        num_neighbors=[32, 8],
                                        input_nodes=("product", graph['product']['node_id']),
                                        batch_size=512,
                                        num_workers=2,
                                        shuffle=False)

    category_nodeloader = NeighborLoader(graph,
                                         num_neighbors=[32, 8],
                                         input_nodes=("category", graph['category']['node_id']),
                                         batch_size=512,
                                         num_workers=2,
                                         shuffle=False)

    model.eval()
    with torch.no_grad(), torch.amp.autocast(device):
        all_embeddings = {node_type: torch.empty(graph[node_type].num_nodes, out_channels) for node_type in
                          ["product", "category"]}

        for batch in tqdm(product_nodeloader):
            # Get embeddings for the current batch's subgraph
            batch_embeddings = model.inference(batch.to(device))

            # Iterate through each node type in the batch
            for node_type, embeddings in batch_embeddings.items():
                if node_type in ["product"]:
                    # Get the global node IDs for this batch's root nodes
                    root_node_ids = batch[node_type].n_id[:batch[node_type].batch_size]

                    # Store the embeddings of the root nodes
                    all_embeddings[node_type][root_node_ids] = embeddings[:batch[node_type]["batch_size"]].cpu()

        for batch in tqdm(category_nodeloader):
            # Get embeddings for the current batch's subgraph
            batch_embeddings = model.inference(batch.to(device))

            # Iterate through each node type in the batch
            for node_type, embeddings in batch_embeddings.items():
                if node_type in ["category"]:
                    # Get the global node IDs for this batch's root nodes
                    root_node_ids = batch[node_type].n_id[:batch[node_type].batch_size]

                    # Store the embeddings of the root nodes
                    all_embeddings[node_type][root_node_ids] = embeddings[:batch[node_type]["batch_size"]].cpu()

    torch.save(all_embeddings, "data/graph/trained_embeddings.pt")
    print("saved embeddings")

if __name__ == "__main__":
    main()

