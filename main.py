from sklearn import tree

X = [[0, 0], [1, 1]]
Y = [0, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

print(clf.predict([[0.499, 0.499]]))
# burada her bir sınıfa ait olasalık döndürür
print(clf.predict_proba([[2, 2]]))