from sklearn import datasets

#Toy dataset containing 1797 observations from images of handwritten digits (good for teaching image classification)
digits = datasets.load_digits()

#Creating a feature matrix
X = digits.data

#Target vector
y = digits.target

#View the 1st observation
print(X[0])