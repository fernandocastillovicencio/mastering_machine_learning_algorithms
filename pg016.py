import numpy as np
import matplotlib.pyplot as plt

# Funções fornecidas
def zero_center(X):
    return X - np.mean(X, axis=0), np.mean(X, axis=0)

def whiten(X):
    Xc, mean = zero_center(X)  # Centralizar os dados
    U, L, Vt = np.linalg.svd(Xc)  # Decomposição SVD
    W = np.dot(Vt.T, np.diag(1.0 / L))  # Matriz de whitening
    X_whitened = np.dot(Xc, W)  # Dados whitened
    return X_whitened, mean, L, Vt

def inverse_whiten(X_whitened, mean, L, Vt):
    W_inv = np.dot(np.diag(L), Vt)  # Operador inverso do whitening
    X_reconstructed = np.dot(X_whitened, W_inv) + mean  # Reconstruir os dados
    return X_reconstructed

# Gerando dados correlacionados
np.random.seed(42)
X = np.random.multivariate_normal(mean=[0, 0], cov=[[2, 1], [1, 1]], size=500)

# Aplicando whitening
X_whitened, mean, L, Vt = whiten(X)

# Reconstruindo os dados originais
X_reconstructed = inverse_whiten(X_whitened, mean, L, Vt)

# Calculando o erro de reconstrução
reconstruction_error = np.linalg.norm(X - X_reconstructed, 'fro')
print(f"Erro de reconstrução: {reconstruction_error:.4e}")

# Plotando o gráfico comparativo
plt.figure(figsize=(18, 6))

# Dados originais
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6, color='orange')
plt.title("Original dataset")
plt.xlabel("X0")
plt.ylabel("X1")
plt.axis("equal")

# Dados whitened
plt.subplot(1, 3, 2)
plt.scatter(X_whitened[:, 0], X_whitened[:, 1], alpha=0.6, color='orange')
plt.title("Whitened dataset")
plt.xlabel("X0")
plt.ylabel("X1")
plt.axis("equal")

# Dados reconstruídos
plt.subplot(1, 3, 3)
plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], alpha=0.6, color='orange')
plt.title("Reconstructed dataset")
plt.xlabel("X0")
plt.ylabel("X1")
plt.axis("equal")

plt.tight_layout()
plt.show()
