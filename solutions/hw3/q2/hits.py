import sys
import numpy as np
from pyspark import SparkConf, SparkContext

ITERATIONS = 40
LAMBDA = 1.0
MU = 1.0

conf = SparkConf()
conf.set('spark.hadoop.validateOutputSpecs', 'false')
sc = SparkContext(conf=conf)

in_path = sys.argv[1]
num_vertices = int(sys.argv[2])

def to_direct_edge(line):
    parts = line.split('\t')
    return (int(parts[0]) - 1, int(parts[1]) - 1)

def to_reverse_edge(line):
    parts = line.split('\t')
    return (int(parts[1]) - 1, int(parts[0]) - 1)

direct_edges = sc.textFile(in_path).map(to_direct_edge).groupByKey()
reverse_edges = sc.textFile(in_path).map(to_reverse_edge).groupByKey()

authority = None
hubbiness = np.ones(num_vertices)

def mult(entry, vec):
    column, vs = entry
    vs = set(vs)
    result = np.zeros(num_vertices)
    for out_v in vs:
        result[out_v] += vec[column]
    return result

for iteration in range(ITERATIONS):
    print(f"Iteration #{iteration + 1}")
    authority = reverse_edges.map(lambda row: mult(row, hubbiness)).sum()
    authority = authority / np.max(authority) # Scale

    hubbiness = direct_edges.map(lambda row: mult(row, authority)).sum()
    hubbiness = hubbiness / np.max(hubbiness) # Scale

    # print(f"New authority: {authority}")
    # print(f"New hubbiness: {hubbiness}")

std = np.argsort(authority)
min_ids = std[:5]
max_ids = std[-5:]
print(f"Authority: min 5 ids={min_ids + np.ones(5)} vals={authority[min_ids]}")
print(f"Authority: max 5 ids={max_ids + np.ones(5)} vals={authority[max_ids]}")

std = np.argsort(hubbiness)
min_ids = std[:5]
max_ids = std[-5:]
print(f"Hubbiness: min 5 ids={min_ids + np.ones(5)} vals={hubbiness[min_ids]}")
print(f"Hubbiness: max 5 ids={max_ids + np.ones(5)} vals={hubbiness[max_ids]}")

sc.stop()
