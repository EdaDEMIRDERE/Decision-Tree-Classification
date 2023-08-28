from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz

iris = load_iris()
X, y = iris.data, iris.target

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

plt.figure(figsize=(20, 10))
tree.plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=[str(name) for name in iris.target_names])
plt.show()