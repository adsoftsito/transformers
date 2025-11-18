import numpy as np
from collections import defaultdict

# ============================
# 1 — Corpus de entrenamiento
# ============================
corpus = """
yo soy gpt
yo soy humano
hola soy gpt
gpt es modelo
yo soy modelo
"""

# ============================
# 2 — Tokenización
# ============================
tokens = corpus.lower().split()

# construir vocabulario
vocab = sorted(set(tokens))
v2i = {w:i for i,w in enumerate(vocab)}
i2v = {i:w for w,i in v2i.items()}

V = len(vocab)

# ============================
# 3 — Matriz de conteos bigrama
# ============================
counts = np.zeros((V, V), dtype=int)

for a, b in zip(tokens[:-1], tokens[1:]):
    counts[v2i[a], v2i[b]] += 1

# ============================
# 4 — Convertir a probabilidades
# ============================
probs = counts / (counts.sum(axis=1, keepdims=True) + 1e-9)

# ============================
# 5 — Función para predecir
# ============================
def predict_next(word):
    if word not in v2i:
        return None
    p = probs[v2i[word]]
    return i2v[np.argmax(p)]   # palabra con mayor probabilidad

# ============================
# 6 — Generar texto
# ============================
def generate(start, length=10):
    out = [start]
    w = start
    for _ in range(length):
        if w not in v2i:
            break
        w = predict_next(w)
        out.append(w)
    return " ".join(out)

# ============================
# Pruebas
# ============================
print("Vocabulario:", vocab)
print("Siguiente después de 'yo' →", predict_next("yo"))
print("Siguiente después de 'soy' →", predict_next("soy"))
print("\nTexto generado:")
print(generate("yo", length=8))

