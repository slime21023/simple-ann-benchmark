import numpy as np 
import time
import pandas as pd
import os  
import faiss

faiss.omp_set_num_threads(8)

mnist = np.load("dataset/mnist-784-euclidean.npy")
print(f'mnist data shape: {mnist.shape}')


# Try to build and run the index

d = mnist.shape[1]
M = 64

HNSW = faiss.IndexHNSWFlat(d, M)
HNSW.hnsw.efConstruction = 64
HNSW.add(mnist)

D, I = HNSW.search(mnist[40].reshape(1, mnist.shape[1]), 100)
print(f'index query: {I}')

flat = faiss.IndexFlatL2(mnist.shape[1])
flat.add(mnist)

D, FLAT_I = flat.search(mnist[40].reshape(1, mnist.shape[1]), k=100) 
print(f'brute query: {FLAT_I}')

# Calculate the recall

print(f"recall: {sum([1 for i in I if i in FLAT_I]) / FLAT_I.size}")
del HNSW

# Run the Benchmark

M = [ 16, 24, 32, 48, 64, 128, 256]
result = []
build_time=[]

def build_index(data, M):
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = 64
    index.add(data)
    return index

def recall(pred, true):
    return sum([1 for i in pred if i in true]) / true.size

def benchmark_knn_query(data, index, size=1000, k=100):

    indices = np.random.choice(data.shape[0], size, replace=False)
    query_time = 0
    cur_recall = 0

    # query
    for i in indices:
        start = time.time()
        D, I = index.search(data[i].reshape(1, data.shape[1]), k=k)
        query_time += (time.time() - start)
        D, FLAT_I = flat.search(data[i].reshape(1, data.shape[1]), k=k) 
        cur_recall += recall(I.flatten(), FLAT_I.flatten())
        
    result.append((query_time/1000, cur_recall/1000))

for m in M:
    start = time.time()
    index = build_index(mnist, M=m)
    btime = time.time() - start
    build_time.append(btime)
    benchmark_knn_query(mnist, index)
    del index

print(f"build time : {build_time}")
print(f"result : {result}")

df = pd.DataFrame(result, columns=['query_time', 'recall'])
df['QPS'] = 1 / df['query_time']
df['build_time'] = build_time
df.plot(x='recall', y='QPS',style='.-')


os.makedirs('results', exist_ok=True)
df.to_csv('results/hnsw-mnist.csv')