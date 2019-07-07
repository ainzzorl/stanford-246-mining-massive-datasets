# Authors: Jessica Su, Wanzi Zhou, Pratyaksh Sharma, Dylan Liu, Ansh Shukla

import numpy as np
import random
import time
import pdb
import unittest
from PIL import Image
import matplotlib.pyplot as plt

# Finds the L1 distance between two vectors
# u and v are 1-dimensional np.array objects
# TODO: Implement this
def l1(u, v):
    return np.linalg.norm(u - v, 1)

# Loads the data into a np array, where each row corresponds to
# an image patch -- this step is sort of slow.
# Each row in the data is an image, and there are 400 columns.
def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')

# Creates a hash function from a list of dimensions and thresholds.
def create_function(dimensions, thresholds):
    def f(v):
        boolarray = [v[dimensions[i]] >= thresholds[i] for i in range(len(dimensions))]
        return "".join(map(str, map(int, boolarray)))
    return f

# Creates the LSH functions (functions that compute L K-bit hash keys).
# Each function selects k dimensions (i.e. column indices of the image matrix)
# at random, and then chooses a random threshold for each dimension, between 0 and
# 255.  For any image, if its value on a given dimension is greater than or equal to
# the randomly chosen threshold, we set that bit to 1.  Each hash function returns
# a length-k bit string of the form "0101010001101001...", and the L hash functions
# will produce L such bit strings for each image.
def create_functions(k, L, num_dimensions=400, min_threshold=0, max_threshold=255):
    functions = []
    for i in range(L):
        dimensions = np.random.randint(low = 0,
                                   high = num_dimensions,
                                   size = k)
        thresholds = np.random.randint(low = min_threshold,
                                   high = max_threshold + 1,
                                   size = k)

        functions.append(create_function(dimensions, thresholds))
    return functions

# Hashes an individual vector (i.e. image).  This produces an array with L
# entries, where each entry is a string of k bits.
def hash_vector(functions, v):
    return np.array([f(v) for f in functions])

# Hashes the data in A, where each row is a datapoint, using the L
# functions in "functions."
def hash_data(functions, A):
    return np.array(list(map(lambda v: hash_vector(functions, v), A)))

# Retrieve all of the points that hash to one of the same buckets
# as the query point.  Do not do any random sampling (unlike what the first
# part of this problem prescribes).
# Don't retrieve a point if it is the same point as the query point.
def get_candidates(hashed_A, hashed_point, query_index):
    return filter(lambda i: i != query_index and \
        any(hashed_point == hashed_A[i]), range(len(hashed_A)))

# Sets up the LSH.  You should try to call this function as few times as
# possible, since it is expensive.
# A: The dataset.
# Return the LSH functions and hashed data structure.
def lsh_setup(A, k = 24, L = 10):
    functions = create_functions(k = k, L = L)
    hashed_A = hash_data(functions, A)
    return (functions, hashed_A)

# Run the entire LSH algorithm
def lsh_search(A, hashed_A, functions, query_index, num_neighbors = 10):
    hashed_point = hash_vector(functions, A[query_index, :])
    candidate_row_nums = get_candidates(hashed_A, hashed_point, query_index)

    distances = map(lambda r: (r, l1(A[r], A[query_index])), candidate_row_nums)
    best_neighbors = sorted(distances, key=lambda t: t[1])[:num_neighbors]

    return [t[0] for t in best_neighbors]

# Plots images at the specified rows and saves them each to files.
def plot(A, row_nums, base_filename):
    for row_num in row_nums:
        patch = np.reshape(A[row_num, :], [20, 20])
        im = Image.fromarray(patch)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(base_filename + "-" + str(row_num) + ".png")

# Finds the nearest neighbors to a given vector, using linear search.
def linear_search(A, query_index, num_neighbors):
    target = A[query_index, :]
    distances = np.apply_along_axis(lambda p: l1(p, target), 1, A)
    nearest_indexes = np.argpartition(distances, num_neighbors + 1)[:(num_neighbors + 1)]

    # Exclude the target point
    nearest_indexes = [i for i in nearest_indexes if i != query_index]
    if len(nearest_indexes) > num_neighbors:
        nearest_indexes = nearest_indexes[:num_neighbors]

    # argpartition does not guarantee sorted order
    nearest_indexes = sorted(nearest_indexes, key=lambda x: (distances[x], x))

    return nearest_indexes

def error(approx_distances, true_distances):
    return np.sum(approx_distances) / np.sum(true_distances)

def problem4():
    data = load_data('./remote/hw1/q4/data/patches.csv')

    compare(data, k=24, L=10, verbose=True)
    plot_errors(data)
    plot_neighbors(data)

def plot_neighbors(data):
    print("Plotting neighbors")
    functions, hashed_data = lsh_setup(data)
    target_index = 100

    linear_res = linear_search(data, target_index, 10)
    lsh_res = lsh_search(data, hashed_data, functions, target_index, 10)

    plot(data, linear_res, './remote/hw1/q4/data//neighbors/linear/neighbor-')
    plot(data, lsh_res, './remote/hw1/q4/data//neighbors/lsh/neighbor-')
    plot(data, [target_index], './remote/hw1/q4/data//neighbors/target-')

def plot_errors(data):
    ls = [10, 12, 14, 16, 18, 20]
    print("Calculating error of L")
    error_of_l = [compare(data, 24, l) for l in ls]
    print("Plotting error of L")
    plt.plot(ls, error_of_l)
    plt.ylabel('Error')
    plt.xlabel('L')
    plt.show()

    ks = [16, 18, 20, 22, 24]
    print("Calculating error of L")
    error_of_k = [compare(data, k, 10) for k in ks]
    print("Plotting error of K")
    plt.plot(ks, error_of_k)
    plt.ylabel('Error')
    plt.xlabel('k')
    plt.show()

def compare(data, k, L, verbose = False):
    print(f"Evaluating data for k={k}, L={L}")
    functions, hashed_data = lsh_setup(data, k=k, L=L)

    num_searches = 10

    total_linear_time = 0.0
    total_lsh_time = 0.0
    total_error = 0.0

    for i in range(1, num_searches + 1):
        target = data[i * 100, :]

        start = time.time()
        linear_res = linear_search(data, i * 100, 3)
        total_linear_time += time.time() - start

        start = time.time()
        lsh_res = lsh_search(data, hashed_data, functions, i * 100, 3)
        if len(lsh_res) < 3:
            print("Returned not enough results, retrying...")
            return compare(data, k, L, verbose)
        total_lsh_time += time.time() - start

        distf = np.vectorize(lambda j: l1(data[j, :], target))

        linear_distances = sorted(distf(linear_res))
        lsh_distances = sorted(distf(lsh_res))

        # if not np.array_equal(linear_distances, lsh_distances) and verbose:
        #     print(f"Linear search and LSH search result mismatch!")
        #     print("Linear:")
        #     print(linear_distances)
        #     print("LSH:")
        #     print(lsh_distances)

        row_error = error(lsh_distances, linear_distances)
        if row_error < 1 and verbose:
            print(f"Error less than 1: {row_error}")
            exit(1)
        total_error += row_error

    avg_error = total_error / num_searches
    if verbose:
        print(f"Average linear search time: {(total_linear_time / num_searches):.2f}")
        print(f"Average LSH search time: {(total_lsh_time / num_searches):.2f}")
        print(f"Average error: {avg_error}")
    return avg_error

#### TESTS #####

class TestLSH(unittest.TestCase):
    def test_l1(self):
        u = np.array([1, 2, 3, 4])
        v = np.array([2, 3, 2, 3])
        self.assertEqual(l1(u, v), 4)

    def test_hash_data(self):
        f1 = lambda v: sum(v)
        f2 = lambda v: sum([x * x for x in v])
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(f1(A[0,:]), 6)
        self.assertEqual(f2(A[0,:]), 14)

        functions = [f1, f2]
        self.assertTrue(np.array_equal(hash_vector(functions, A[0, :]), np.array([6, 14])))
        self.assertTrue(np.array_equal(hash_data(functions, A), np.array([[6, 14], [15, 77]])))

    ### TODO: Write your tests here (they won't be graded,
    ### but you may find them helpful)


if __name__ == '__main__':
    #unittest.main() ### TODO: Uncomment this to run tests
    problem4()
