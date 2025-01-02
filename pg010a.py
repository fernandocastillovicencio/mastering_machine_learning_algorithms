import numpy as np

# @profile
def multivariate():
    nb_samples = 200
    mu = [1.0, 1.0]
    covm = [[2.0, 0.0], [0.0, 0.8]]

    X =  np.random.multivariate_normal(mean=mu, cov=covm, size=nb_samples)
    return X

def main():
    X = multivariate()
    print(X)
    
if __name__ == "__main__":
    main()