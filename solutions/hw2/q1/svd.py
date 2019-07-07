from scipy import linalg
import numpy as np

M = np.array([1, 2, 2, 1, 3, 4, 4, 3]).reshape((-1, 2))
print(f"M={M}")

u, sigma, vt = linalg.svd(M, full_matrices=False)
print(f"U={u} sigma={sigma} VT={vt}")

mtm = np.transpose(M).dot(M)
print(f"MTM={mtm}")

evals, evecs = linalg.eigh(mtm)
#print(f"evals={evals}, evecs={evecs}")
si = np.argsort(-evals)
evals = evals[si]
evecs = evecs[:, si]
print(f"evals={evals}, evecs={evecs}")
