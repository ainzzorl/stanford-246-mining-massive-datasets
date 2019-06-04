import sys
import math
import matplotlib.pyplot as plt
import numpy as np

# TODO: create custom iterator

K = 20
NUM_ITERATIONS = 40
LAMBDA = 0.1

if len(sys.argv) < 2:
    print(f"Usage: python recommended.py PATH_TO_DATA_DIR")
    exit(1)

DATA_DIR = sys.argv[1]
IN_FILE = f"{DATA_DIR}/ratings.train.txt"

NUM_USERS = 0
NUM_MOVIES = 0

def parse(l):
    p = l.rstrip('\n').split('\t')
    return (int(p[0]), int(p[1]), int(p[2]))

with open(IN_FILE) as f:
    for l in f:
        d = parse(l)
        NUM_USERS = max(NUM_USERS, d[0] + 1)
        NUM_MOVIES = max(NUM_MOVIES, d[1] + 1)

print(f"Num users: {NUM_USERS}, num movies: {NUM_MOVIES}")

def init():
    q = np.random.rand(NUM_MOVIES, K) * math.sqrt(5.0 / K)
    p = np.random.rand(K, NUM_USERS) * math.sqrt(5.0 / K)
    pt = np.transpose(p)
    return (q, p, pt)

def run_iteration(lr):
    with open(IN_FILE) as f:
        for l in f:
            user, movie, rating = parse(l)
            pred_rating = np.dot(Q[movie], PT[user])
            if np.isnan(pred_rating):
                print("Predicted rating not a number!")
                return 200000.0

            e = 2 * (rating - pred_rating)
            dq = - e * PT[user] + 2 * LAMBDA * Q[movie]
            dp = - e * Q[movie] + 2 * LAMBDA * PT[user]
            Q[movie] -= lr * dq
            PT[user] -= lr * dp
    total_error = 0.0
    with open(IN_FILE) as f:
        for l in f:
            user, movie, actual_rating = parse(l)
            predicted_rating = np.dot(Q[movie], PT[user])
            total_error += (actual_rating - predicted_rating) ** 2
    for i in range(NUM_MOVIES):
        total_error += LAMBDA * (np.linalg.norm(Q[i], 2) ** 2)
    for i in range(NUM_USERS):
        total_error += LAMBDA * (np.linalg.norm(PT[i], 2) ** 2)
    return total_error

LRS = [0.001, 0.01, 0.05, 0.1]
COLORS = ['red', 'green', 'blue', 'orange']
lr_errors = {}
for lr in LRS:
    print(f"LR={lr}")

    # Q=Movies, P=Users
    Q, P, PT = init()

    lr_errors[lr] = []
    for iteration in range(NUM_ITERATIONS):
        print(f"Iteration: {iteration}")
        iteration_error = run_iteration(lr)
        lr_errors[lr].append(iteration_error)
        print(f"Error: {iteration_error}")
    print()

plots = []
legends = []
for cnt, lr in enumerate(LRS):
    p, = plt.plot(range(1, NUM_ITERATIONS + 1), lr_errors[lr], COLORS[cnt])
    plots.append(p)
    legends.append(f"lr={lr}")
plt.legend(plots, legends, loc=1)

plt.ylabel('Error')
plt.xlabel('Iteration')
plt.title('SGD')
plt.savefig(f'{DATA_DIR}/plot.png')
plt.close()
