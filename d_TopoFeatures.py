



import os
import pandas as pd
import networkx as nx

datadir = 'data_spider'

# Read data.
df_net_allsample = pd.read_csv(os.path.join(datadir, '_metaspider_allsample.csv'))
# Set the first two columns as character type, and the following columns as floating-point number type.
df_net_allsample.iloc[:, 0:2] = df_net_allsample.iloc[:, 0:2].astype(str)
df_net_allsample.iloc[:, 2:] = df_net_allsample.iloc[:, 2:].astype(float)

# Select the species (node) column and the weight column.
species_columns = df_net_allsample.columns[:2]
weight_columns = df_net_allsample.columns[2:]

# Extract species name.
species_list = df_net_allsample[species_columns].values.flatten()
# Remove duplicates and generate a list.
species_list = list(set(species_list))

list_A = []
list_B = []
list_C = []
# Traverse each sample.
for sample_column in weight_columns:
    # print(f"Sample: {sample_column}")
    # Create an empty undirected graph
    G = nx.Graph()

    # Add edges and weights to the graph
    for _, row in df_net_allsample.iterrows():
        species1 = row[species_columns[0]]
        species2 = row[species_columns[1]]
        weight = row[sample_column]
        # If weight is not nan, add edge and weight
        if pd.notnull(weight):
            G.add_edge(species1, species2, weight=weight)

    # Calculate weighted degrees for each node
    weighted_degrees = dict(G.degree(weight='weight'))
    weighted_degrees = pd.DataFrame(list(weighted_degrees.items()), columns=['Node', f'{sample_column}'])
    # Calculate weighted clustering coefficients for each node
    weighted_clustering = nx.clustering(G, weight='weight')
    weighted_clustering = pd.DataFrame(list(weighted_clustering.items()), columns=['Node', f'{sample_column}'])
    # Calculate centralities for each node
    centralities = nx.betweenness_centrality(G)
    centralities = pd.DataFrame(list(centralities.items()), columns=['Node', f'{sample_column}'])

    list_A.append(weighted_degrees)
    list_B.append(weighted_clustering)
    list_C.append(centralities)


# Get the topological metrics for all nodes in all samples
df_merged = pd.merge(list_A[0], list_A[1], on=['Node'], how='outer')
for i in range(2, len(list_A)):
    df_merged = pd.merge(df_merged, list_A[i], on=['Node'], how='outer')
df_weighted_degrees = df_merged.copy()
df_weighted_degrees.to_csv(os.path.join(datadir, '_topo_weighted_degrees.csv'), index=False)

df_merged = pd.merge(list_B[0], list_B[1], on=['Node'], how='outer')
for i in range(2, len(list_B)):
    df_merged = pd.merge(df_merged, list_B[i], on=['Node'], how='outer')
df_weighted_clustering = df_merged.copy()
df_weighted_clustering.to_csv(os.path.join(datadir, '_topo_weighted_clustering.csv'), index=False)

df_merged = pd.merge(list_C[0], list_C[1], on=['Node'], how='outer')
for i in range(2, len(list_C)):
    df_merged = pd.merge(df_merged, list_C[i], on=['Node'], how='outer')
df_centralities = df_merged.copy()
df_centralities.to_csv(os.path.join(datadir, '_topo_centralities.csv'), index=False)


