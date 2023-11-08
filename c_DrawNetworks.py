



import os
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg') # Set up a non-interactive backend !
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

datadir = 'data_spider'

'''
- The first two columns of df_net_allsample are' feature_x' and' feature_y', followed by the corresponding sample name.
- 'feature_x' and 'feature_y' represent two nodes, the value in each sample represents the edge between the two nodes, 
and nan represents that there is no such edge.
- Draw a network diagram for each sample, including the weight values of nodes and edges.
'''

# Read data.
df_net_allsample = pd.read_csv(os.path.join(datadir, '_metaspider_allsample.csv'))
network_path = os.path.join(datadir, 'network')
if not os.path.exists(network_path):
    os.makedirs(network_path)

def process(i, network_path, df_net_allsample):
    '''
    Realize the Shell layout algorithm, the thickness of the edge reflects the absolute value of the weight,
    and the color of the edge reflects positive (blue) and negative (red).
    '''
    samplename = df_net_allsample.columns[i]
    # Create an empty graphic object.
    G = nx.Graph()
    # Add nodes.
    G.add_nodes_from(df_net_allsample[['feature_x', 'feature_y']].values.flatten())
    # Add edges and their weight values.
    for index, row in df_net_allsample.iterrows():
        if not pd.isna(row[i]):
            G.add_edge(row['feature_x'], row['feature_y'], weight=row[i])

    # Draw network diagram.
    pos = nx.shell_layout(G)  # Using the shell_layout algorithm.
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    edge_colors = ['blue' if weight >= 0 else 'red' for weight in weights]
    edge_widths = [abs(weight) * 2 for weight in weights]

    plt.figure(figsize=(15, 15))
    # Draw edges.
    for i, (u, v, d) in enumerate(G.edges(data=True)):
        weight = weights[i]
        edge_color = edge_colors[i]
        width = edge_widths[i]
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=edge_color, width=width, alpha=0.5)
    # Draw nodes.
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color='lightblue')
    # Draw labels.
    nx.draw_networkx_labels(G, pos, font_color='black', font_size=8)

    plt.title(f'Meta-Spider Network Graph - Sample {samplename}', fontsize='xx-large')
    plt.axis('off')

    # Save network diagram
    # plt.savefig(os.path.join(network_path, f'{samplename}.png'), dpi=300)
    plt.savefig(os.path.join(network_path, f'{samplename}.pdf'), dpi=300)
    plt.close()

# Use multithreading to draw each network diagram.
Parallel(n_jobs=8)(delayed(process)(i, network_path, df_net_allsample) for i in range(2, len(df_net_allsample.columns)))
