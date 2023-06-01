import numpy as np 
import time
from annoy import AnnoyIndex
import pandas as pd
import os  
import faiss

faiss.omp_set_num_threads(8)

mnist = np.load("dataset/mnist-784-euclidean.npy")
print(f'mnist data shape: {mnist.shape}')

index = AnnoyIndex(f=mnist.shape[1], metric='euclidean')

for i in range(mnist.shape[0]):
    index.add_item(i, vector=mnist[i, :])

index.build(n_trees=100)

I = index.get_nns_by_vector(vector=mnist[0], n=100)
print(f'index query: {I}')

flat = faiss.IndexFlatL2(mnist.shape[1])
flat.add(mnist)

D, FLAT_I = flat.search(mnist[0].reshape(1, mnist.shape[1]), k=100) 
print(f'brute query: {FLAT_I}')

# Calculate the recall

print(f"recall: {sum([1 for i in I if i in FLAT_I]) / FLAT_I.size}")
del index

# Run the Benchmark

tree_nums = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
# tree_nums = [1, 10, 30]
build_time=[]
result = []

def build_index(data, n_trees, metric='euclidean'):
    index = AnnoyIndex(f=data.shape[1], metric='euclidean')
    for i in range(data.shape[0]):
        index.add_item(i, vector=data[i, :])

    index.build(n_trees)
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
        qk = index.get_nns_by_vector(vector=data[i], n=k)
        query_time += (time.time() - start)
        D, FLAT_I = flat.search(data[i].reshape(1, data.shape[1]), k=k) 
        cur_recall += recall(qk, FLAT_I)
    
    result.append((query_time/1000, cur_recall/1000))


for t_num in tree_nums:
    start = time.time()
    index = build_index(mnist, n_trees=t_num)
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
df.to_csv('results/annoy-mnist.csv')