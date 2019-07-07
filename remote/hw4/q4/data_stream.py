import sys
import math
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['agg.path.chunksize'] = 10000

EPS = 0.0001 * math.exp(1)
DELTA = math.exp(-5)

N_HASHES = math.ceil(math.log(1 / DELTA))
N_BUCKETS = math.ceil(math.exp(1) / EPS)

stream_path = sys.argv[1]
true_counts_path = sys.argv[2]
hash_params_path = sys.argv[3]
stream_length = int(sys.argv[4])

#Returns hash(x) for hash function given by parameters a, b, p and n_buckets
def hash_fun(a, b, p, n_buckets, x):
    y = x % p
    hash_val = (a*y + b) % p
    return hash_val % n_buckets

def read_hash_params():
    result = []
    with open(hash_params_path) as f:
        for line in f.readlines():
            result.append([int(x) for x in line.split()])
    return result

print('Reading hash params')
hash_params = read_hash_params()

counts = np.zeros((N_HASHES, N_BUCKETS))

def process_stream():
    with open(stream_path) as f:
        for line in f.readlines():
            num = int(line)
            for hid in range(N_HASHES):
                hv = hash_fun(hash_params[hid][0], hash_params[hid][1], 123457, N_BUCKETS, num)
                counts[hid][hv] += 1

print('Processing stream')
process_stream()

plot_xs = []
plot_ys = []

print('Calculating errors')
with open(true_counts_path) as tcf:
    for tc_line in tcf.readlines():
        parts = [int(x) for x in tc_line.split()]
        word_id = parts[0]
        true_count = parts[1]

        per_hash_counts = [counts[hid, hash_fun(hash_params[hid][0], hash_params[hid][1], 123457, N_BUCKETS, word_id)] for hid in range(N_HASHES)]
        estimate_count = min(per_hash_counts)
        error = 1.0 * (estimate_count - true_count) / true_count
        freq = 1.0 * true_count / stream_length
        #print(f"{word_id} actual={true_count} estimate={estimate_count} error={error}")

        plot_xs.append(freq)
        plot_ys.append(error)

print('Plotting')
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.plot(plot_xs, plot_ys, color='blue', lw=2)
ax.set_xlabel('True frequency')
ax.set_ylabel('Error')
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()
