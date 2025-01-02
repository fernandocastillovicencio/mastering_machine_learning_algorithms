import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import describe
import pandas as pd

from sklearn.preprocessing import Normalizer


# @profile
def generate(nb_samples, mu, covm):
    X =  np.random.multivariate_normal(mean=mu, cov=covm, size=nb_samples)
    return X


def summary(data):
    df = pd.DataFrame(data)
    print(f"Dimensões: {df.shape}\n")
    return df.describe()


def main():   
    nb_samples = 200
    mu = [1.0, 1.0]
    covm = [[2.0, 0.0], [0.0, 0.8]]
    X = generate(nb_samples, mu, covm)

    
    nz = Normalizer(norm='l2')
    X_nz = nz.fit_transform(X)

    print(summary(X))
    print(summary(X_nz))
    
    plt.scatter(X_nz[:,0], X_nz[:,1])
    plt.grid()
    plt.axis('equal')
    plt.show()
    
if __name__ == "__main__":
    main()