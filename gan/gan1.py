import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ===== Parámetros iniciales =====
x_real = 2.0
z = 1.0

w_g = 0.5
b_g = 0.0
w_d = 1.0
b_d = 0.0

lr = 0.1
ITERACIONES = 2

print("===== ENTRENAMIENTO MINI-GAN =====")
print("Todo coincide exactamente con las fórmulas de Excel\n")

for i in range(1, ITERACIONES + 1):
    print(f"\n--- Iteración {i} ---")

    # ===== Paso 1: Generador produce dato falso =====
    x_fake = w_g * z + b_g

    # ===== Paso 2: Discriminador evalúa =====
    D_real = sigmoid(w_d * x_real + b_d)
    D_fake = sigmoid(w_d * x_fake + b_d)

    # ===== Paso 3: Pérdidas =====
    loss_D = -(np.log(D_real) + np.log(1 - D_fake))
    loss_G = -np.log(D_fake)

    # ===== Paso 4: Gradientes del Discriminador =====
    dL_dw_d = -((1 - D_real) * x_real - (D_fake * x_fake))
    dL_db_d = -((1 - D_real) - (-D_fake))

    # ===== Actualizar Discriminador =====
    w_d_new = w_d - lr * dL_dw_d
    b_d_new = b_d - lr * dL_db_d

    # ===== Paso 5: Gradientes del Generador =====
    dL_dw_g = -(1 - D_fake) * w_d * z
    dL_db_g = -(1 - D_fake) * w_d

    # ===== Actualizar Generador =====
    w_g_new = w_g - lr * dL_dw_g
    b_g_new = b_g - lr * dL_db_g

    # ===== Imprimir resultados =====
    print(f"Fake data: {x_fake:.6f}")
    print(f"D(real):  {D_real:.6f}")
    print(f"D(fake):  {D_fake:.6f}")
    print(f"Loss_D:   {loss_D:.6f}")
    print(f"Loss_G:   {loss_G:.6f}")
    print(f"Grad_D (dw, db): {dL_dw_d:.6f}, {dL_db_d:.6f}")
    print(f"Grad_G (dw, db): {dL_dw_g:.6f}, {dL_db_g:.6f}")
    print(f"Pesos D -> w_d={w_d_new:.6f}, b_d={b_d_new:.6f}")
    print(f"Pesos G -> w_g={w_g_new:.6f}, b_g={b_g_new:.6f}")

    # ===== Preparar siguiente iteración =====
    w_d, b_d = w_d_new, b_d_new
    w_g, b_g = w_g_new, b_g_new

