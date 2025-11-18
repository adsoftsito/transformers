import numpy as np

# ========================================
# 1) Embeddings manuales
# ========================================
E = {
    "yo": np.array([1.0, 0.0]),
    "soy": np.array([0.0, 1.0])
}

# Secuencia de entrada
tokens = ["yo", "soy"]

X = np.stack([E[t] for t in tokens])     # shape (2, 2)
# X =
# [[1,0],
#  [0,1]]

# ========================================
# 2) Positional Encoding (mínimo)
# ========================================
PE = np.array([
    [0.0, 0.1],   # posición 0
    [0.1, 0.0]    # posición 1
])

X = X + PE
# X ahora =
# [[1.0, 0.1],
#  [0.1, 1.0]]

# ========================================
# 3) Parámetros del Self-Attention
# (muy pequeños y manuales)
# ========================================
W_Q = np.array([[1.0, 0.0],
                [0.0, 1.0]])

W_K = np.array([[1.0, 0.0],
                [0.0, 1.0]])

W_V = np.array([[1.0, 0.0],
                [0.0, 1.0]])

# ========================================
# 4) Cálculo de Q, K, V
# ========================================
Q = X @ W_Q     # shape (2,2)
K = X @ W_K
V = X @ W_V

# ========================================
# 5) Atención Escalar (Softmax muy simple)
# ========================================
scores = Q @ K.T            # (2,2)
scores = scores / np.sqrt(2)  # scale

# Softmax fila por fila
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

attn = np.vstack([softmax(scores[i]) for i in range(2)])

# ========================================
# 6) Combinación con valores V
# ========================================
Z = attn @ V   # salida del self-attention

# ========================================
# 7) Mini Feed-Forward Network
# ========================================
W1 = np.array([[1.0, 0.5],
               [0.5, 1.0]])

b1 = np.array([0.1, 0.1])

ff = Z @ W1 + b1

# ========================================
# 8) Resultados
# ========================================
print("Embeddings + posiciones:\n", X)
print("\nScores:\n", scores)
print("\nAtención (softmax):\n", attn)
print("\nSalida Self-Attention:\n", Z)
print("\nSalida FeedForward:\n", ff)

