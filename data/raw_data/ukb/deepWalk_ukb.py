import networkx as nx
from node2vec import Node2Vec

# Load the hyperedges data
def load_hyperedges(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    edges = [list(map(int, line.strip().split(','))) for line in lines]
    return edges

# Create a graph from the hyperedges
def create_graph(hyperedges):
    G = nx.Graph()
    for edge in hyperedges:
        for i in range(len(edge)):
            for j in range(i + 1, len(edge)):
                G.add_edge(edge[i], edge[j])
    return G

# Load hyperedges
hyperedges = load_hyperedges('/Users/wenyuanhuizi/Desktop/TACCO/data/raw_data/ukb/hyperedges-ukb.txt')

# Create graph
G = create_graph(hyperedges)

# Generate walks using Node2Vec
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)

# Learn embeddings
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Save embeddings
embeddings = model.wv
embeddings.save_word2vec_format('node_embeddings_ukb.txt')