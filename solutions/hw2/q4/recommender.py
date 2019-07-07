import sys
import numpy as np

if len(sys.argv) < 2:
    print(f"Usage: python recommended.py PATH_TO_DATA_DIR")
    exit(1)

DATA_DIR = sys.argv[1]

shows = []
with open(f"{DATA_DIR}/shows.txt") as f:
    for l in f:
        shows.append(l)
shows = np.array(shows)

R = np.loadtxt(f"{DATA_DIR}/user-shows.txt")
m = len(R)
n = len(R[0])
print(f"n={n} m={m}")

def pow_minus_one_half(x):
    if x == 0:
        return 0
    else:
        return pow(x, -0.5)

vf = np.vectorize(pow_minus_one_half)

P_minus_one_half = np.diag(vf(np.sum(R, axis=1)))
print(f"P shape={P_minus_one_half.shape}, expected: {m}x{m}")

Q_minus_one_half = np.diag(vf(np.sum(R, axis=0)))
print(f"Q shape={Q_minus_one_half.shape}, expected: {n}x{n}")

def recommend(gamma):
    # TODO: Break ties by taking the shows with smaller index
    scores = gamma[499,][:100]
    idx = np.argsort(scores)[-5:]
    print(shows[idx])

print(f"Building movie-movie collab filter...")
gamma_movies = R
gamma_movies = np.matmul(gamma_movies, Q_minus_one_half)
gamma_movies = np.matmul(gamma_movies, np.transpose(R))
gamma_movies = np.matmul(gamma_movies, R)
gamma_movies = np.matmul(gamma_movies, Q_minus_one_half)

print("Movie-movie recommendations:")
recommend(gamma_movies)

print(f"Building user-user collab filter...")
gamma_users = P_minus_one_half
gamma_users = np.matmul(gamma_users, R)
gamma_users = np.matmul(gamma_users, np.transpose(R))
gamma_users = np.matmul(gamma_users, P_minus_one_half)
gamma_users = np.matmul(gamma_users, R)

print("User-user recommendations:")
recommend(gamma_users)
