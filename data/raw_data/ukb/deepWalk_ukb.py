from gensim.models import Word2Vec
import networkx as nx
from random import choice

# Function to perform deep walks
def deepwalk(G, walk_length, num_walks):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                cur = walk[-1]
                neighbors = list(G.neighbors(cur))
                if len(neighbors) > 0:
                    walk.append(choice(neighbors))
                else:
                    break
            walks.append(walk)
    return walks

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
hyperedges = load_hyperedges('data/raw_data/ukb/hyperedges-ukb.txt')

# Create graph
G = create_graph(hyperedges)

# Perform deep walks
walks = deepwalk(G, walk_length=10, num_walks=50)

# Train Word2Vec model
model = Word2Vec(walks, vector_size=32, window=10, min_count=1, sg=1, workers=8)

# Save embeddings
model.wv.save_word2vec_format('node_embeddings_ukb.txt')
