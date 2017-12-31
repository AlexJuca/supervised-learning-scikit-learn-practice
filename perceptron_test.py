import pandas as pd
import numpy as np
import perceptron as pcn 

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

ppn = pcn.Perceptron(alpha=0.1, n_iterations=10)
ppn.fit(X, y)
