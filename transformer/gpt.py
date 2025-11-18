import numpy as np

np.random.seed(0)

# ===============================
# Datos
# ===============================
vocab = ["yo", "soy", "gpt", "hola", "fin"]
v2i = {w:i for i,w in enumerate(vocab)}
i2v = {i:w for w,i in v2i.items()}

train_data = [
    ["yo", "soy", "gpt"],
    ["yo", "soy", "hola"],
    ["hola", "soy", "gpt"]
]

# Convertir secuencia → índices → matrices one-hot
def seq_to_indices(seq):
    return np.array([v2i[w] for w in seq], dtype=int)

# ===============================
# Hiperparámetros
# ===============================
d_model = 3
seq_len = 3
vocab_size = len(vocab)
lr = 0.1
epochs = 200

# ===============================
# Parámetros del Transformer
# ===============================
E = np.random.randn(vocab_size, d_model) * 0.1      # embedding

W_Q = np.random.randn(d_model, d_model) * 0.1
W_K = np.random.randn(d_model, d_model) * 0.1
W_V = np.random.randn(d_model, d_model) * 0.1

W1 = np.random.randn(d_model, d_model) * 0.1        # feed-forward
b1 = np.zeros((d_model,))

W_out = np.random.randn(d_model, vocab_size) * 0.1
b_out = np.zeros((vocab_size,))

# ===============================
# Funciones auxiliares
# ===============================
def softmax(x):
    x2 = x - np.max(x)
    e = np.exp(x2)
    return e / e.sum(axis=-1, keepdims=True)

def one_hot(idx, size):
    v = np.zeros(size)
    v[idx] = 1
    return v

# ===============================
# Entrenamiento
# ===============================
for epoch in range(epochs):

    loss_sum = 0

    for seq in train_data:

        idx = seq_to_indices(seq)
        
        # ========= FORWARD =========
        X = E[idx]                           # embeddings (3,3)
        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V

        scores = (Q @ K.T) / np.sqrt(d_model)   # (3,3)
        A = softmax(scores)                     # atención

        Z = A @ V                                # (3,3)

        H = Z @ W1 + b1                          # feed-forward

        logits = H @ W_out + b_out               # salida
        probs = softmax(logits)                  # probas predicción

        # expectativa: predecir palabra siguiente
        targets = np.array([one_hot(i, vocab_size) for i in idx])

        loss = -np.sum(targets * np.log(probs + 1e-9))
        loss_sum += loss

        # ========= BACKWARD =========
        dlogits = probs - targets                # derivada CE

        dW_out = H.T @ dlogits
        db_out = dlogits.sum(axis=0)

        dH = dlogits @ W_out.T

        dW1 = Z.T @ dH
        db1 = dH.sum(axis=0)

        dZ = dH @ W1.T

        dA = dZ @ V.T
        dV = A.T @ dZ

        dscores = dA * A - A * np.sum(dA * A, axis=-1, keepdims=True)

        dQ = dscores @ K
        dK = dscores.T @ Q

        dW_Q = X.T @ dQ
        dW_K = X.T @ dK
        dW_V = X.T @ dV

        dX = dQ @ W_Q.T + dK @ W_K.T + dV @ W_V.T

        dE = np.zeros_like(E)
        for i_tok, idx_tok in enumerate(idx):
            dE[idx_tok] += dX[i_tok]

        # ========= UPDATE =========
        W_out -= lr * dW_out
        b_out -= lr * db_out
        W1 -= lr * dW1
        b1 -= lr * db1
        W_Q -= lr * dW_Q
        W_K -= lr * dW_K
        W_V -= lr * dW_V
        E -= lr * dE

    # cada 20 épocas
    if epoch % 20 == 0:
        print(f"Epoch {epoch}, loss = {loss_sum:.4f}")

# ===============================
# PRUEBA
# ===============================
def predict_next(seq):
    idx = seq_to_indices(seq)

    X = E[idx]
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    A = softmax((Q @ K.T)/np.sqrt(d_model))
    Z = A @ V
    H = Z @ W1 + b1
    logits = H @ W_out + b_out
    probs = softmax(logits)

    return i2v[np.argmax(probs[-1])]

print("\n=== Prueba ===")
print("Entrada: ['yo','soy'] →", predict_next(["yo","soy","yo"]) )
print("Entrada: ['hola','soy'] →", predict_next(["hola","soy","yo"]) )

