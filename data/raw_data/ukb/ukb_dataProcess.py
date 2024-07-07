import numpy as np

# Function to read the dataset and find unique medical codes
def get_unique_medical_codes(filename):
    unique_codes = set()
    with open(filename, 'r') as file:
        for line in file:
            # Split the line by commas and convert each code to an integer
            codes = list(map(int, line.strip().split(',')))
            unique_codes.update(codes)
    return list(unique_codes)

# File name of the dataset
#filename = '/Users/wenyuanhuizi/Desktop/TACCO/data/raw_data/ukb/hyperedges-ukb.txt'
filename = 'hyperedges-ukb.txt'

# Get the unique medical codes
unique_medical_codes = get_unique_medical_codes(filename)
num_unique_codes = len(unique_medical_codes)

print(f"Number of unique medical codes: {num_unique_codes}")

# Number of unique medical codes
num_nodes = num_unique_codes

# Dimensionality of the embeddings
embedding_dim = 64  # You can choose this based on your requirements

# Function to randomly initialize embeddings
def random_initialization(num_nodes, embedding_dim):
    embeddings = np.random.rand(num_nodes, embedding_dim)
    return embeddings

# Randomly initialize embeddings
embeddings = random_initialization(num_nodes, embedding_dim)

# Save the embeddings to a file for later use
np.savetxt('random_embeddings.txt', embeddings)

print("Randomly initialized embeddings:")
print(embeddings)

