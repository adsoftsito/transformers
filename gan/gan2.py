import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# === Dato real: una "imagen" 3x3 ===

text = [1, 1, 1, 1, 0, 1, 1, 1, 1] 


X_real = np.array(text).reshape(3,3)
print(X_real)
X_real = X_real.reshape(9)
print(X_real)

#X_real = np.eye(3).reshape(9)   # vector de 9 elementos

# X_real(2) = 1


# === Ruido para G ===
z = np.ones(9)

# === Pesos iniciales ===
rng = np.random.default_rng(0)

Wg = rng.normal(0, 0.1, (9, 9))
bg = np.zeros(9)

Wd = rng.normal(0, 0.1, 9)
bd = 0.0

lr = 0.02
ITER = 2

print("======= ENTRENANDO MINI-GAN 3x3 =======")

for i in range(1, ITER + 1):

    # ---- Generador produce matriz falsa ----
    X_fake = Wg @ z + bg  # vector de 9
    X_fake_img = X_fake.reshape(3, 3)

    # ---- Discriminador evalúa ----
    D_real = sigmoid(Wd @ X_real + bd)
    D_fake = sigmoid(Wd @ X_fake + bd)

    # ---- Pérdidas ----
    loss_D = -(np.log(D_real) + np.log(1 - D_fake))
    loss_G = -np.log(D_fake)

    # ---- Gradientes Discriminador ----
    dL_dWd = -( (1 - D_real)*X_real - D_fake*X_fake )
    dL_dbd = -( (1 - D_real) - D_fake )

    # ---- Actualizar Discriminador ----
    Wd -= lr * dL_dWd
    bd -= lr * dL_dbd

    # ---- Gradientes Generador ----
    # D_fake depende de Wd @ X_fake, y X_fake = Wg @ z + bg
    dL_dWg = -(1 - D_fake) * np.outer(Wd, z)
    dL_dbg = -(1 - D_fake) * Wd

    # ---- Actualizar Generador ----
    Wg -= lr * dL_dWg
    bg -= lr * dL_dbg

    # ---- Mostrar ----
    print(f"\n--- Iteración {i} ---")
    print("Matriz generada 3x3:")
    print(X_fake_img)
    for r in range(3):
        print(int(round(X_fake_img[r,0],0)), int(round(X_fake_img[r,1],0)), int(round(X_fake_img[r,2],0)))
    print(f"D(real) = {D_real:.4f}, D(fake) = {D_fake:.4f}")
    print(f"Loss_D = {loss_D:.4f}, Loss_G = {loss_G:.4f}")

