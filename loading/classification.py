from sklearn.datasets import make_classification

# Generate features matrix and target vector
X, y = make_classification(n_samples = 100,
                           n_features = 3,
                           n_informative = 3,
                           n_redundant = 0,
                           n_classes = 2,
                           weights = [.25, .75],
                           random_state = 1)

# View feature matrix and target vector
print('Feature Matrix\n', X[:3])
print('Target Vector\n', y[:3])
