from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

voting_clf = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(random_state=42)),
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(random_state=42)),
       
         ('dtree',DecisionTreeClassifier(random_state=42)),
         ('gauss',GaussianNB()),
         ('k',KNeighborsClassifier())
    ]
)
voting_clf.named_estimators['svc'].probability = True

print(y_test)
voting_clf.voting='soft'
voting_clf.fit(X_train, y_train)

predictions = voting_clf.predict(X_test)

accuracy = accuracy_score(predictions, y_test)
precision= precision_score(predictions, y_test)
recall= recall_score(predictions, y_test)
f1 = f1_score(predictions,y_test)
print(accuracy)
print(precision)
print(recall)
print(f1)