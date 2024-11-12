from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris = load_iris(as_frame=True)

#print(iris)

X_iris = iris.data[["petal length (cm)","petal width (cm)"]].values
y_iris = iris.target
tree_clf = DecisionTreeClassifier(max_depth=2,random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris , test_size=0.25, random_state=0)

tree_clf.fit(X_train,y_train)