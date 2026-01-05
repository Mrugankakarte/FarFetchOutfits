from logging import exception

import networkx as nx
import plotly.graph_objects as go
from torch_geometric.data import HeteroData
from torch_geometric.utils import to_networkx
from collections import defaultdict


def create_plotly_graph_visualization(pyg_graph: HeteroData):
    """
    Creates a rich, interactive Plotly visualization for a HeteroData object.

    This function handles different node and edge types, assigning them
    distinct colors and creating a comprehensive legend. It displays the
    actual node IDs on the plot and correctly identifies supervised edges.

    Args:
        pyg_graph (HeteroData): The PyG heterogeneous graph to visualize.
                                It's assumed that each node type has an `n_id`
                                attribute holding the original node identifiers.
    """
    # --- 1. Convert to NetworkX and get node positions ---
    # Pass node attributes to preserve them. We need 'type' for coloring
    # and 'n_id' for labeling.
    G = to_networkx(pyg_graph, node_attrs=['n_id'], edge_attrs=['e_id'], to_undirected=False)

    # Use a layout algorithm to find the position for each node.
    # A seed ensures the layout is reproducible.
    pos = nx.spring_layout(G, k=3, iterations=500, seed=42)
    #pos = nx.kamada_kawai_layout(G)
    # --- 2. Create a lookup for supervised edge labels ---
    n_id_to_nx_id = {data['n_id']: node for node, data in G.nodes(data=True) if 'n_id' in data}
    label_lookup = {}

    for et in pyg_graph.edge_types:
        if hasattr(pyg_graph[et], 'edge_label_index'):
            edge_label_index = pyg_graph[et].edge_label_index
            src_type, _, dst_type = et

            # Local node ids to original node ids
            src_n_ids = pyg_graph[src_type].n_id
            dst_n_ids = pyg_graph[dst_type].n_id

            # Also get edge labels if they exist to build the lookup
            edge_labels = pyg_graph[et].edge_label if hasattr(pyg_graph[et], 'edge_label') else None

            for i in range(edge_label_index.size(1)):
                src_local_idx = edge_label_index[0, i].item()
                dst_local_idx = edge_label_index[1, i].item()

                # Map local indices to original n_ids
                src_n_id = src_n_ids[src_local_idx].item()
                dst_n_id = dst_n_ids[dst_local_idx].item()

                # Find the corresponding NetworkX node IDs
                u = n_id_to_nx_id.get(src_n_id)
                v = n_id_to_nx_id.get(dst_n_id)

                # Add the edge to the graph if both nodes exist in the sample
                # and the edge doesn't already exist from message-passing.
                if u is not None and v is not None and not G.has_edge(u, v):
                    G.add_edge(u, v, type=et, e_id='Supervised')

                # Populate the label lookup for hover text
                if edge_labels is not None:
                    label = edge_labels[i].item()
                    label_lookup[(src_n_id, dst_n_id)] = label
    print(label_lookup)

    # --- 3. Define Color Schemes ---
    node_type_colors = {
        "family": "#4599C3",  # Blue
        "category": "#ED8546",  # Orange
        "product": "#70B349",  # Green
    }
    default_node_color = "#999999"

    edge_type_colors = {
        ('family', 'fam_outfit', 'family'): "#8B4D9E",  # Purple
        ('category', 'cat_outfit', 'category'): "#DFB825",  # Gold
        ('product', 'product_outfit', 'product'): "#DB5C64",  # Red
    }
    default_edge_color = "#888888"

    # --- 4. Create Traces for Edges ---
    edge_traces = []
    edges_by_type = defaultdict(list)
    for u, v, data in G.edges(data=True):
        edge_type = data.get('type')
        edges_by_type[edge_type].append((u, v, data))

    for edge_type, edges in edges_by_type.items():
        edge_x, edge_y = [], []
        mid_x, mid_y = [], []
        hover_texts = []

        for u, v, data in edges:
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            mid_x.append((x0 + x1) / 2)
            mid_y.append((y0 + y1) / 2)

            # Get actual node IDs for hover text
            u_id = G.nodes[u].get('n_id', u)
            v_id = G.nodes[v].get('n_id', v)
            e_id = data.get('e_id', 'N/A')
            hover_text = (
                f"<b>{edge_type[1]}</b><br>"
                f"Edge ID: {e_id}<br>"
                f"{G.nodes[u]['type']} {u_id} → {G.nodes[v]['type']} {v_id}"
            )

            if (u_id, v_id) in label_lookup:
                label_val = label_lookup[(u_id, v_id)]
                label_str = "Positive" if label_val == 1 else "Negative"
                hover_text += f"<br><b>Label: {label_str} (Training Edge)</b>"
            hover_texts.append(hover_text)

        # Trace for the edge lines
        line_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color=edge_type_colors.get(edge_type, default_edge_color)),
            hoverinfo='none',
            mode='lines',
            name=f"{edge_type[0]} → {edge_type[2]} ({edge_type[1]})",
            legendgroup="Edges"
        )
        edge_traces.append(line_trace)

        # Create a trace for the hover markers and on-plot labels
        label_trace = go.Scatter(
            x=mid_x, y=mid_y,
            mode='markers+text',
            text=[edge_type[1]] * len(mid_x),  # Display relation type as label
            textposition="top center",
            textfont=dict(size=8, color='#444'),
            marker=dict(size=5, color=edge_type_colors.get(edge_type, default_edge_color), opacity=0.6),
            hovertext=hover_texts,
            hoverinfo='text',
            showlegend=False,
        )
        edge_traces.append(label_trace)

    # --- 5. Create Traces for Nodes ---
    node_traces = []
    nodes_by_type = defaultdict(list)
    for node, data in G.nodes(data=True):
        node_type = data.get('type')
        nodes_by_type[node_type].append(node)

    for node_type, nodes in nodes_by_type.items():
        node_x, node_y, hover_texts, node_labels = [], [], [], []
        for node in nodes:
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Use the 'n_id' attribute for labels, fall back to the index if not found
            node_id = G.nodes[node].get('n_id', node)
            hover_texts.append(f"Type: {node_type}<br>ID: {node_id}")
            node_labels.append(f"{node_id}")  # Display actual ID on the node

        trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_labels,
            textposition="middle center",
            textfont=dict(color='white', size=10),
            hoverinfo='text',
            hovertext=hover_texts,
            marker=dict(
                color=node_type_colors.get(node_type, default_node_color),
                size=35,
                line=dict(width=2, color='black')
            ),
            name=node_type,
            legendgroup="Nodes"
        )
        node_traces.append(trace)

    # --- 6. Assemble and Display the Figure ---
    fig = go.Figure(data=edge_traces + node_traces,
                    layout=go.Layout(
                        width=1200,
                        height=900,
                        title=dict(
                            text='<b>Interactive Heterogeneous Graph Visualization</b>',
                            font=dict(size=20)
                        ),
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        legend=dict(
                            title="<b>Legend</b>",
                            itemsizing='constant',
                            traceorder='grouped'
                        )
                    ))

    return fig


from train import Model
import numpy as np
from numpy.random import permutation
from itertools import combinations
import pandas as pd
import torch
import random
import time
from tqdm.auto import tqdm


def precision_k(y_true, y_pred, k=12):
    x = np.intersect1d(y_true, y_pred[:k])
    return len(x) / k


def relevence_k(y_true, y_pred, k=12):
    if y_pred[k - 1] == y_true:
        return 1
    else:
        return 0


def avg_precision_k(y_true, y_pred, k=12):
    ap = 0.0
    n = len(y_pred)
    k = min(n, k)
    for i in range(1, k + 1):
        prec = precision_k(y_true, y_pred, i)
        ap += prec * relevence_k(y_true, y_pred, i)

    return ap / min(1, k)


def mean_avg_precision_k(y_true, y_pred, k=12):
    return np.round(np.mean([avg_precision_k(yt, yp, k) for yt, yp in zip(y_true, y_pred)]), 6)


def compute_hr(df, k=1):
    df['label'] = df.apply(lambda x: x['missing_prod_nodeid'] in x['ranked_candidates'][:k], axis=1)
    return sum(df['label']) / df.shape[0]


class ValidationDataGenerator(torch.utils.data.Dataset):
    def _candidate_generation(self, x):
        missing_prod_category = x['missing_prod_category']
        if self.missing_candidates_from_same_cat:
            candidates = list(set(self.product_df[self.product_df['product_category'] == missing_prod_category][
                                      'product_node_id'].values) - set([x['missing_prod_nodeid']]))
            # missing_prod_id_list = torch.tensor(list(self.product_df[self.product_df['product_category']==missing_prod_category]['product_node_id']))
        else:
            candidates = list(set(self.product_df.product_node_id.values) - set(x['incomplete_outfits_nodeids']) - set(
                [x['missing_prod_nodeid']]))

        candidates = random.sample(candidates, min(self.candidate_len, len(candidates)))
        candidates = candidates + [x['missing_prod_nodeid']]
        random.shuffle(candidates)
        return candidates

    def __init__(self, validation_outfits_path, products_path, sample_outfit_length=None,
                 missing_candidates_from_same_cat=True, candidate_len=5):

        self.candidate_len = candidate_len
        self.missing_candidates_from_same_cat = missing_candidates_from_same_cat

        self.validation_outfits = pd.read_parquet(validation_outfits_path)
        if sample_outfit_length is None:
            sample_outfit_length = self.validation_outfits.shape[0]

        self.validation_outfits = self.validation_outfits.sample(sample_outfit_length).reset_index(drop=True)
        self.product_df = pd.read_parquet(products_path)
        print(f"validating on {self.validation_outfits.shape[0]} outfits")

        self.all_products_nodeids = torch.tensor(self.product_df['product_node_id'].to_list())
        self.nodeid_to_productid_map = self.product_df.set_index("product_node_id")['product_id'].to_dict()
        self.productid_to_nodeid_map = self.product_df.set_index("product_id")['product_node_id'].to_dict()
        self.product_category_map = self.product_df.set_index('product_id')['product_category'].to_dict()

        self.validation_outfits["products_shuffled"] = self.validation_outfits.apply(
            lambda row: permutation(row["products"]).tolist(), axis=1)
        self.validation_outfits["incomplete_outfit"] = self.validation_outfits.apply(
            lambda row: row["products_shuffled"][:-1], axis=1)
        self.validation_outfits["missing_product"] = self.validation_outfits.apply(
            lambda row: row["products_shuffled"][-1], axis=1)

        self.validation_outfits['incomplete_outfits_nodeids'] = self.validation_outfits['incomplete_outfit'].apply(
            lambda x: [self.productid_to_nodeid_map[i] for i in x])
        self.validation_outfits['missing_prod_nodeid'] = self.validation_outfits['missing_product'].map(
            self.productid_to_nodeid_map)
        self.validation_outfits['missing_prod_category'] = self.validation_outfits['missing_product'].map(
            self.product_category_map)

        print(f"generating candidates")
        t1 = time.time()
        self.validation_outfits['candidates'] = self.validation_outfits.progress_apply(
            lambda x: self._candidate_generation(x), axis=1)
        print(f"completed in {time.time() - t1} s")

    def __len__(self):
        return self.validation_outfits.shape[0]

    def __getitem__(self, idx):
        # print(idx)

        '''
        missing_prod_category = self.validation_outfits['missing_prod_category'][idx]
        if self.missing_candidates_from_same_cat:
            candidates = list(set(self.product_df[self.product_df['product_category']==missing_prod_category]['product_node_id'].values)-set([self.validation_outfits['missing_prod_nodeid'][idx]]))
            #missing_prod_id_list = torch.tensor(list(self.product_df[self.product_df['product_category']==missing_prod_category]['product_node_id']))
        else:
            candidates = list(set(self.product_df.product_node_id.values)-set(self.validation_outfits['incomplete_outfits_nodeids'][idx])-set([self.validation_outfits['missing_prod_nodeid'][idx]]))

        candidates = random.sample(candidates, self.candidate_len)
        candidates = candidates + [self.validation_outfits['missing_prod_nodeid'][idx]]
        '''
        missing_prod_id_list = torch.tensor(self.validation_outfits['candidates'][idx])

        incomplete_outfit = torch.tensor(self.validation_outfits['incomplete_outfits_nodeids'][idx])
        incomplete_outfit = torch.tile(incomplete_outfit, dims=(len(missing_prod_id_list), 1))
        incomplete_graphs = torch.column_stack((incomplete_outfit, missing_prod_id_list))
        # print(incomplete_graphs)
        return incomplete_graphs, missing_prod_id_list

def load_model(graph, num_node_dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = "cpu"
    print(f"DEVICE:{device}")
    out_channels = 64
    model = Model(in_channels=64, hidden_channels=64, out_channels=out_channels, heads=8,
                  num_nodes_dict=num_node_dict, graph_metadata=graph.metadata(),
                  layer_type='GAT', device=device)
    try:
        model.load_state_dict(torch.load("model_weights.pt", weights_only=True, map_location="cpu"))
        print("loaded model weights...")
    except exception as e:
        print(f"ERROR:{e}")
        pass
    model.to(device)

    print(model)
    print(f"model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return model

def get_fitb_score(model, outfit, all_embeddings):
    score = []
    for i, j in combinations(outfit, 2):
        u_feats = all_embeddings[i]
        v_feats = all_embeddings[j]
        pscore = model.pred_layer(u_feats * v_feats)
        score.append(pscore)
    return torch.sum(torch.stack(score))


def main():

    graph = torch.load("data/graph/graph.pt", weights_only=False)
    print(graph)

    num_node_dict = {}

    for node_type in graph.node_types:
        num_node_dict[node_type] = graph[node_type].num_nodes

    print(num_node_dict)

    model = load_model(graph, num_node_dict)

    all_embeddings = torch.load("data/graph/trained_embeddings.pt", weights_only=False, map_location="cpu")
    print("loaded embeddings")

    valid_gen = ValidationDataGenerator(validation_outfits_path="data/testing_outfits.parquet",
                                        products_path="data/graph/products.parquet",
                                        sample_outfit_length=10000,
                                        missing_candidates_from_same_cat=False, candidate_len=12
                                        )

    vdl = torch.utils.data.DataLoader(valid_gen, batch_size=1, shuffle=False)

    val_outfit_scores = []
    with torch.no_grad():
        for i in tqdm(vdl):
            scores = []
            missing_prod_list = i[1][0]
            for j in i[0][0]:
                # print(torch.tensor(j))
                scores.append(get_fitb_score(model, j, all_embeddings['product']))

            k = min(len(scores), 10)
            topk_prods = torch.topk(torch.tensor(scores), k=k)
            pids = topk_prods.indices

            # print(topk_prods.values)
            # print(topk_prods.indices)
            # print(missing_prod_list[topk_prods.indices])
            val_outfit_scores.append(missing_prod_list[pids])

    sample_valid = valid_gen.validation_outfits
    sample_valid['ranked_candidates'] = [i.numpy() for i in val_outfit_scores]
    sample_valid

    # candidate prods from same category
    k = [15, 12, 11, 10, 5, 3, 2, 1]

    for i in k:
        print(
            f"MAP@{i}: {mean_avg_precision_k(sample_valid['missing_prod_nodeid'].tolist(), sample_valid['ranked_candidates'].tolist(), k=i)}")

    for i in k:
        print(f"HitRate@{i}:{compute_hr(sample_valid, i)}")

    sample_valid.to_parquet("ranked_samples_notsamecat.parquet")

if __name__ == "__main__":
    main()