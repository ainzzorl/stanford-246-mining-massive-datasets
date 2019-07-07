import sys
import numpy as np
from pyspark import SparkConf, SparkContext

ITERATIONS = 40
BETA = 0.8

conf = SparkConf()
conf.set('spark.hadoop.validateOutputSpecs', 'false')
sc = SparkContext(conf=conf)

in_path = sys.argv[1]
num_vertices = int(sys.argv[2])

def to_edge(line):
    parts = line.split('\t')
    return (int(parts[0]) - 1, int(parts[1]) - 1)

edges = sc.textFile(in_path).map(to_edge).groupByKey()

pagerank = np.ones(num_vertices) / num_vertices

def mult(entry, pr):
    column, out_edges = entry
    out_edges = set(out_edges)
    degree = len(out_edges)
    result = np.zeros(num_vertices)
    for out_dest in out_edges:
        result[out_dest] += pr[column]
    result *= BETA / degree
    return result

for iteration in range(ITERATIONS):
    print(f"Iteration #{iteration + 1}")
    map_out = edges.map(lambda row: mult(row, pagerank))
    pagerank = map_out.sum() + np.ones(num_vertices) * (1.0 - BETA) / num_vertices
    #print(f"New pagerank: {pagerank}")

std = np.argsort(pagerank)
min_ids = std[:5]
max_ids = std[-5:]
print(f"Min 5 ids={min_ids} vals={pagerank[min_ids]}")
print(f"Max 5 ids={max_ids} vals={pagerank[max_ids]}")
sc.stop()
