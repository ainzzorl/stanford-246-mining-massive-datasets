import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from pyspark import SparkConf, SparkContext

MAX_ITER = 20
NUM_CLUSTERS = 10

conf = SparkConf()
conf.set('spark.hadoop.validateOutputSpecs', 'false')
sc = SparkContext(conf=conf)

data_path = sys.argv[1]
initial_centroids_random_path = sys.argv[2]
initial_centroids_far_apart_path = sys.argv[3]
out_path = sys.argv[4]

def to_point(line):
    return np.array([float(p) for p in line.split(' ')])

data = sc.textFile(data_path).map(to_point)

DIMENSIONS = len(data.first())

current_centroids_broadcasted = None
norm_broadcasted = None
error_accumulator = None

with open(initial_centroids_random_path) as f:
    initial_centroids_random = np.array([np.array(to_point(l)) for l in f])
with open(initial_centroids_far_apart_path) as f:
    initial_centroids_far_apart = np.array([np.array(to_point(l)) for l in f])

# Returns (cluster id, point) tuple
def find_nearest_centroid(point):
    cc = current_centroids_broadcasted.value
    distances_to_centroids = np.apply_along_axis(lambda c: np.linalg.norm(point - c, ord=norm_broadcasted.value), 1, cc)
    cluster_id = np.argmin(distances_to_centroids)
    point_error = distances_to_centroids[cluster_id]
    error_accumulator.add(point_error)

    return (np.argmin(distances_to_centroids), point)

def run_clustering(norm, initial_centroids):
    global current_centroids_broadcasted, norm_broadcasted, error_accumulator
    errors = []

    current_centroids = np.array(initial_centroids)
    for iteration in range(1, MAX_ITER + 1):
        print(f"Iteration #{iteration}")
        current_centroids_broadcasted = sc.broadcast(current_centroids)
        norm_broadcasted = sc.broadcast(norm)
        error_accumulator = sc.accumulator(0.0)
        cluster_id_to_point = data.map(find_nearest_centroid)

        agg = (0, np.zeros(DIMENSIONS))
        aggregated = cluster_id_to_point.aggregateByKey(agg, lambda a, b: (a[0] + 1, a[1] + b), lambda a, b: (a[0] + b[0], a[1] + b[1]))
        avgs = aggregated.map(lambda v: v[1][1]/v[1][0])

        current_centroids = np.array(avgs.collect())

        error = error_accumulator.value
        errors.append(error)
    return errors

if not os.path.exists(out_path):
    os.makedirs(out_path)

errors = {}
norms = [1, 2]
for norm in norms:
    errors[norm] = {}
    for initial_centroids in [(initial_centroids_random, 'random'), (initial_centroids_far_apart, 'far apart')]:
        cv = initial_centroids[0]
        cn = initial_centroids[1]
        print(f"Norm - {norm}, initial centroids - {cn}")
        errors[norm][cn] = run_clustering(norm, cv)

print(errors)

for norm in norms:
    p1, = plt.plot(range(1, MAX_ITER + 1), errors[norm]['random'], 'red')
    p2, = plt.plot(range(1, MAX_ITER + 1), errors[norm]['far apart'], 'blue')
    legend = plt.legend([p1, p2], ["Random Initial Cluster", "Far Apart Initial Clusters"], loc=1)
    plt.ylabel('Error')
    plt.xlabel('Iteration')
    plt.title(f'k-means error with norm={norm}')
    plt.savefig(f'{out_path}/l{norm}-plot.png')
    plt.close()

sc.stop()
