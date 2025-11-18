import numpy as np
from scipy.special import softmax

M = np.array([[1,0,1],
              [0,1,1]])    # memory patterns as columns

q = np.array([1,0.2])       # query


scores = M.T @ q            # similarity
weights = softmax(scores)   # attention weights
output = M @ weights        # retrieved vector

print("Scores:", scores)
print("Weights:", weights)
print("Output:", output)

