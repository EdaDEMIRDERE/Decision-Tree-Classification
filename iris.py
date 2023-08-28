from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz

iris = load_iris()
X, y = iris.data, iris.target

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)