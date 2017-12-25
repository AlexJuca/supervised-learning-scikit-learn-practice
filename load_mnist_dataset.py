from sklearn import datasets
import matplotlib.pyplot as pyplot

# Load MNIST dataset
digits = datasets.load_digits()

# print the description and keys of the mnist datasets
print(digits.DESCR)
print(digits.keys)

# print the shape of the images and data
print(digits.images.shape)
print(digits.data.shape)

# display digit 10
pyplot.imshow(digits.images[10], cmap=pyplot.cm.gray_r, interpolation='nearest')
pyplot.show()

