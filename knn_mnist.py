from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# load the mnist dataset
digits = datasets.load_digits()

# create feature and target arrays
X = digits.data
y = digits.target

# split data into training and test
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create a KNN classifier with 7 neighbors
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier model to the training data
knn.fit(X_train, y_train)

# print the accuracy of the classifier using the test data
print(knn.score(X_test, y_test))
