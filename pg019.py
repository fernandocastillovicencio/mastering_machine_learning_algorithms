from sklearn.model_selection import train_test_split


nb_samples = 200
mu = [1.0, 1.0]
covm = [[2.0, 0.0], [0.0, 0.8]]

X =  np.random.multivariate_normal(mean=mu, cov=covm, size=nb_samples)



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_
size=0.7, random_state=1000)