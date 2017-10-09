from sklearn.datasets import make_regression

X, y, coef = make_regression(n_samples=100,
                             n_features=3,
                             n_informative=3, #number of features used to generate the target value
                             n_targets=1,
                             noise=0.0,
                             coef=True,
                             random_state=1)

# View feature matrix and target vector
print('Feature Matrix\n', X[:3])
print('Target Vector\n', y[:3])