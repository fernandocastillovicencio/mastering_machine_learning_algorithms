import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

# Parâmetros
nb_samples = 200
mu = [1.0, 1.0]
covm = [[2.0, 0.0], [0.0, 0.8]]

# Geração dos dados
X = np.random.multivariate_normal(mean=mu, cov=covm, size=nb_samples)

# Normalização
nz = Normalizer(norm='l2')
Y = nz.transform(X)

# Criação de subplots
fig, ax = plt.subplots(1,2, figsize=(8, 10))  # Dois gráficos, empilhados verticalmente

# Gráfico 1: Dados originais
ax[0].scatter(X[:, 0], X[:, 1], label="Original", alpha=0.5)
ax[0].set_title("Dados Originais")
ax[0].legend()

# Gráfico 2: Dados Normalizados
ax[1].scatter(Y[:, 0], Y[:, 1], label="Normalizado", alpha=0.5, color='orange')
ax[1].set_title("Dados Normalizados")
ax[1].legend()

# Ajustes finais
fig.tight_layout()
plt.show()