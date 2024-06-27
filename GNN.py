import matplotlib.pyplot as plt
from math import isclose
from sklearn.decomposition import PCA
import os
import networkx as nx
import numpy as np
import pandas as pd
from stellargraph import StellarGraph, datasets
from stellargraph.data import EdgeSplitter
from collections import Counter
import multiprocessing
from IPython.display import display, HTML
from sklearn.model_selection import train_test_split
import stellargraph as sg
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import FullBatchLinkGenerator
from stellargraph.layer import GCN, LinkEmbedding


from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection

from stellargraph import globalvar

data_url = "./Dataset/Cit-HepPh.txt"
df_data_1 = pd.read_csv(data_url, sep='\t', skiprows=4, names=['FromNodeId', 'ToNodeId'], dtype={'FromNodeId': int, 'ToNodeId': int})


data_url = "./Dataset/cit-HepPh-dates.txt"
df_data_2 = pd.read_csv(data_url, sep='\t', skiprows=1, names=['NodeId', 'Date'], dtype={'NodeId': str, 'Date': str})
df_data_2['Date'] = pd.to_datetime(df_data_2['Date'])
df_data_2 = df_data_2[~df_data_2['NodeId'].str.startswith('11')]
df_data_2['NodeId'] = df_data_2['NodeId'].astype(str).str.lstrip('0')
df_data_2['NodeId'] = df_data_2['NodeId'].astype(int)
df_data_2 = df_data_2[df_data_2['Date'].dt.year <= 1993]

df_merged = pd.merge(df_data_1, df_data_2, how='inner', left_on='FromNodeId', right_on='NodeId')
df_merged['Date'] = pd.to_datetime(df_merged['Date'])
# Filter out rows where 'ToNodeId' is not present in 'NodeId' column of df_data_2
df_merged = df_merged[df_merged['ToNodeId'].isin(df_data_2['NodeId'])]
# Rename columns 'from' and 'to' to 'source' and 'target' respectively
df_merged.rename(columns={'FromNodeId': 'source', 'ToNodeId': 'target'}, inplace=True)
# unnodes = df_merged['FromNodeId'].unique()
# i = 0
# for nodes in unnodes:
#     i += 1
# print(i)
# Get the number of rows in the DataFrame
num_rows = len(df_merged)

# Concatenate unique nodes from 'source' and 'target' columns
unique_nodes = pd.concat([df_merged['source'], df_merged['target']]).drop_duplicates()

# Create a DataFrame with a constant feature for each unique node
node_features_df = pd.DataFrame({'constant_feature': [1] * len(unique_nodes)}, index=unique_nodes, columns=['constant_feature'])


G = StellarGraph(node_features_df, df_merged)

print(G.info())


# Define an edge splitter on the original graph G:
edge_splitter_test = EdgeSplitter(G)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
# reduced graph G_test with the sampled links removed:
G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global", keep_connected=True
)

# Define an edge splitter on the reduced graph G_test:
edge_splitter_train = EdgeSplitter(G_test)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
# reduced graph G_train with the sampled links removed:
G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
    p=0.1, method="global", keep_connected=True
)

epochs = 50

train_gen = FullBatchLinkGenerator(G_train, method="gcn")
train_flow = train_gen.flow(edge_ids_train, edge_labels_train)

test_gen = FullBatchLinkGenerator(G_test, method="gcn")
test_flow = train_gen.flow(edge_ids_test, edge_labels_test)

gcn = GCN(
    layer_sizes=[16, 16], activations=["relu", "relu"], generator=train_gen, dropout=0.3
)
x_inp, x_out = gcn.in_out_tensors()
prediction = LinkEmbedding(activation="relu", method="ip")(x_out)

prediction = keras.layers.Reshape((-1,))(prediction)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=0.01),
    loss=keras.losses.binary_crossentropy,
    metrics=["acc"],
)

# history = model.fit(
#     train_flow, epochs=epochs, validation_data=test_flow, verbose=2, shuffle=False
# )

# train_metrics = model.evaluate(train_flow)
# test_metrics = model.evaluate(test_flow)

# print("\nTrain Set Metrics of the trained model:")
# for name, val in zip(model.metrics_names, train_metrics):
#     print("\t{}: {:0.4f}".format(name, val))

# print("\nTest Set Metrics of the trained model:")
# for name, val in zip(model.metrics_names, test_metrics):
#     print("\t{}: {:0.4f}".format(name, val))
