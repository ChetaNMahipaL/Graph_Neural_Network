import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
data_url = "./Dataset/Cit-HepPh.txt"
# Load the dataset with specified data types
df = pd.read_csv(data_url, sep='\t', skiprows=4, names=['FromNodeId', 'ToNodeId'], dtype={'FromNodeId': int, 'ToNodeId': int})

# Construct the directed graph
G = nx.from_pandas_edgelist(df, 'FromNodeId', 'ToNodeId', create_using=nx.DiGraph())

# Print basic information about the graph
print("Number of nodes:", len(G.nodes()))
print("Number of edges:", len(G.edges()))


