import numpy as np 
import time
import pandas as pd
import os  
import faiss

faiss.omp_set_num_threads(8)

sift = np.load("dataset/sift-128-euclidean.npy")
print(f'sift data shape: {sift.shape}')


# Try to build and run the index

d = sift.shape[1]

m = 16
nbits = 9
PQ = faiss.IndexPQ(d, m, nbits)
print(PQ.is_trained)
PQ.train(sift)
PQ.add(sift)

D, I = PQ.search(sift[40].reshape(1, sift.shape[1]), 100)
print(f'index query: {I}')

flat = faiss.IndexFlatL2(sift.shape[1])
flat.add(sift)

D, FLAT_I = flat.search(sift[40].reshape(1, sift.shape[1]), k=100) 
print(f'brute query: {FLAT_I}')

# Calculate the recall

print(f"recall: {sum([1 for i in I if i in FLAT_I]) / FLAT_I.size}")
del PQ


# Run the Benchmark

bits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
build_time=[]
result=[]

def build_index(data, n_bits, m=16):
    index = faiss.IndexPQ(data.shape[1], m, n_bits)
    index.train(data)
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


for n_bits in bits:
    start = time.time()
    index = build_index(sift, n_bits=n_bits)
    btime = time.time() - start
    build_time.append(btime)
    benchmark_knn_query(sift, index)
    del index


print(f"build time : {build_time}")
print(f"result : {result}")

df = pd.DataFrame(result, columns=['query_time', 'recall'])
df['QPS'] = 1 / df['query_time']
df['build_time'] = build_time
df.plot(x='recall', y='QPS',style='.-')


os.makedirs('results', exist_ok=True)
df.to_csv('results/pq-sift.csv')