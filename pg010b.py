import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

@profile
def generate(nb_samples, mu, covm):
    X =  np.random.multivariate_normal(mean=mu, cov=covm, size=nb_samples)
    return X

@profile
def plot(X, X_mms, X_ss, X_rs):
    fig,ax = plt.subplots(2, 2, figsize=(10, 8))
    
    # Original dataset
    ax[0, 0].scatter(X[:, 0], X[:, 1])
    ax[0, 0].set_title("Original dataset")
    ax[0, 0].set_xlim(-6, 6)
    ax[0, 0].set_ylim(-6, 6)
    ax[0, 0].grid(True)

    # Min-Max scaling
    ax[0, 1].scatter(X_mms[:, 0], X_mms[:, 1])
    ax[0, 1].set_title("Min-Max scaling (-1, 1)")
    ax[0, 1].set_xlim(-6, 6)
    ax[0, 1].set_ylim(-6, 6)
    ax[0, 1].grid(True)

    # Standard scaling
    ax[1, 0].scatter(X_ss[:, 0], X_ss[:, 1])
    ax[1, 0].set_title(r"Standard scaling ($\mu=0, \sigma=1$)")
    ax[1, 0].set_xlim(-6, 6)
    ax[1, 0].set_ylim(-6, 6)
    ax[1, 0].grid(True)

    # Robust scaling
    ax[1, 1].scatter(X_rs[:, 0], X_rs[:, 1])
    ax[1, 1].set_title("Robust scaling (10th, 90th quantiles)")
    ax[1, 1].set_xlim(-6, 6)
    ax[1, 1].set_ylim(-6, 6)
    ax[1, 1].grid(True)

    
    
    # plt.tight_layout()
    # plt.show()


def main():   
    nb_samples = 200
    mu = [1.0, 1.0]
    covm = [[2.0, 0.0], [0.0, 0.8]]
    X = generate(nb_samples, mu, covm)

    ss = StandardScaler()
    X_ss = ss.fit_transform(X)
    
    rs = RobustScaler(quantile_range=(10,90))
    X_rs = rs.fit_transform(X)
    
    mms = MinMaxScaler(feature_range=(-1,1))
    X_mms = mms.fit_transform(X)

    plot(X, X_mms, X_ss, X_rs)
    
if __name__ == "__main__":
    main()