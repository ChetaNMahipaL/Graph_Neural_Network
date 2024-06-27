# High-Energy Physics Citation Network Analysis

## Project Overview

This project explores and analyzes the Arxiv HEP-PH (High Energy Physics Phenomenology) citation network, a dataset containing 34,546 papers with 421,578 citation edges. The project encompasses graph exploration, community detection, and link prediction tasks, providing insights into the structure and evolution of scientific collaboration in high-energy physics.

Data Source: [SNAP: Stanford Network Analysis Project](http://snap.stanford.edu/data/cit-HepPh.html)

## Key Features

1. Temporal Graph Analysis
2. Community Detection
3. Link Prediction using Graph Neural Networks and Classic Algorithms

## Tasks and Methodologies

### Task 1: Graph Exploration

Analyzed the evolution of the citation network over time, focusing on five key properties:

- Density
- Average In-degree vs Out-degree
- Edges vs Nodes growth
- Shrinking Diameter phenomenon
- Centrality Measures (PageRank, Betweenness, Eigenvector)

Visualizations and metrics are provided to support the findings and showcase the network's structural changes over time.

### Task 2: Community Detection

Implemented and compared two community detection algorithms:

1. Girvan-Newman Algorithm
2. Label Propagation Algorithm

Performed temporal community analysis to study the evolution of research communities over time. The analysis includes:

- Identification of major research clusters
- Temporal changes in community structure
- Visualization of community dynamics

### Task 3: Link Prediction

Implemented and compared two approaches for link prediction:

1. Graph Neural Network (GNN)
2. Node2Vec (classic algorithm)

The models were trained on the citation network up to time T and evaluated on subsequent edges. The comparison includes:

- Performance metrics for both approaches
- Insights into the link prediction task for citation networks

## Technologies Used

- Python
- NetworkX
- PyTorch Geometric
- Matplotlib/Seaborn for visualizations
- Scikit-learn for evaluation metrics

## Key Findings

Please refer to [Analysis](/Analysis.pdf)

## Future Work

- Explore more advanced GNN architectures for link prediction
- Investigate the impact of external features (e.g., paper abstracts) on community detection and link prediction
- Extend the analysis to other scientific domains for comparative studies

## References

- [Node2Vec: Scalable Feature Learning for Networks](https://arxiv.org/pdf/1607.00653.pdf)
- For quick summary and comment on above paper [Click](/Research_Paper.pdf)

## **Execution of code**
### **Task_1**
Install the necessary libraries:
- `pip install networkx`
- `pip install matplotlib`
- `pip install scikit-learn`

Run the jupyter notebook named **Task_1.ipynb**.

### **Task_2**
Install the necessary libraries:
- `pip install networkx`
- `pip install matplotlib`
- `pip install scikit-learn`

Run the jupyter notebook named 
- **Task_2_GN.ipynb** (Contains Community Detection Algorithm-1)
- **Task_2_LabelP.ipynb** (Contains Community Detection Algorithm-2)
- **Task_2 Temporal.ipynb** (Contains Temporal Community Detection)

**All the results are stored in Plots folder for reference**
