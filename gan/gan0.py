import numpy as np

# Datos reales
real_data = np.array([1, 2, 3], dtype=float)

# Inicializacion de parametros
w_g, b_g = 0.5, 0.0  # generador
w_d, b_d = 1.0, 0.0  # discriminador

# Funcion sigmoide
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Funcion generadora
def generator(z):
    return w_g * z + b_g

# Funcion discriminadora
def discriminator(x):
    return sigmoid(w_d * x + b_d)

# Entrada aleatoria para el generador
z = np.array([0.5, 1.0, 1.5])

# Salida del generador
fake_data = generator(z)
print("Fake data:", fake_data)

# Salida del discriminador
D_real = discriminator(real_data)
D_fake = discriminator(fake_data)
print("D(real):", D_real)
print("D(fake):", D_fake)

# Perdida simple de GAN (Binary Cross Entropy)
loss_D = -np.mean(np.log(D_real) + np.log(1 - D_fake))
loss_G = -np.mean(np.log(D_fake))
print("Loss Discriminador:", loss_D)
print("Loss Generdor:", loss_G)

