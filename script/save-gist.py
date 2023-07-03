import numpy as np 
import time
from annoy import AnnoyIndex
import pandas as pd
import os  
import faiss


# faiss.omp_set_num_threads(8)

gist = np.load("../dataset/half-gist-960-euclidean.npy")
print(f'gist data shape: {gist.shape}')

if not os.path.exists('../save'):
   os.makedirs('../save')

# Build Annoy index
tree_nums = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

def build_index(data, n_trees):
    index = AnnoyIndex(f=data.shape[1], metric='euclidean')
    for i in range(data.shape[0]):
        index.add_item(i, vector=data[i, :])

    index.build(n_trees)
    return index


for t_num in tree_nums:
    index = build_index(gist, n_trees=t_num)
    index.save(f'../save/gist-{t_num}.ann')
    del index


#  Build LSH index
d = gist.shape[1]

bits = [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960]

def build_index(data, n_bits):
    index = faiss.IndexLSH(data.shape[1], n_bits)
    index.add(data)
    return index

for n_bits in bits:
    index = build_index(gist, n_bits=n_bits)
    faiss.write_index(index, f"../save/gist-{n_bits}.lsh")
    del index

# Build HNSW index
d = gist.shape[1]

M = [ 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]

def build_index(data, M):
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = 64
    index.add(data)
    return index

for m in M:
    index = build_index(gist, M=m)
    faiss.write_index(index, f"../save/gist-{m}.hnsw")
    del index

