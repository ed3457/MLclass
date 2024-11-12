import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

dataset = load_breast_cancer(as_frame=True)
print(dataset['data'])

X = dataset['data']
y = dataset['target']


X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.25, random_state=0)




ss_train = StandardScaler()
X_train = ss_train.fit_transform(X_train)

ss_test = StandardScaler()
X_test = ss_test.fit_transform(X_test)

#fit computes the mean and stdev to be used for later scaling, note it's just a computation with no scaling done.

#transform uses the previously computed mean and stdev to scale the data (subtract mean from all values and then divide it by stdev).

#fit_transform does both at the same time.



model = LogisticRegression()

model.fit(X_train,y_train)
predictions = model.predict(X_test)



cm = confusion_matrix(y_test, predictions)

TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()

print('True Positive(TP)  = ', TP)
print('False Positive(FP) = ', FP)
print('True Negative(TN)  = ', TN)
print('False Negative(FN) = ', FN)

accuracy =  (TP + TN) / (TP + FP + TN + FN)

#print('Accuracy of the binary classifier = {:0.3f}'.format(accuracy))


models = {}

# Logistic Regression
from sklearn.linear_model import LogisticRegression # type: ignore
models['Logistic Regression'] = LogisticRegression()

# Support Vector Machines
from sklearn.svm import LinearSVC
models['Support Vector Machines'] = LinearSVC()

# Decision Trees
from sklearn.tree import DecisionTreeClassifier
models['Decision Trees'] = DecisionTreeClassifier()

# Random Forest
from sklearn.ensemble import RandomForestClassifier
models['Random Forest'] = RandomForestClassifier()

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
models['Naive Bayes'] = GaussianNB()

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
models['K-Nearest Neighbor'] = KNeighborsClassifier()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy, precision, recall,f1 = {}, {}, {},{}

for key in models.keys():
    
    # Fit the classifier
    models[key].fit(X_train, y_train)
    
    # Make predictions
    predictions = models[key].predict(X_test)
    
    # Calculate metrics
    accuracy[key] = accuracy_score(predictions, y_test)
    precision[key] = precision_score(predictions, y_test)
    recall[key] = recall_score(predictions, y_test)
    f1[key] = f1_score(predictions,y_test)

import pandas as pd

df_model = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall'])
df_model['Accuracy'] = accuracy.values()
df_model['Precision'] = precision.values()
df_model['Recall'] = recall.values()
df_model['F1'] = f1.values()

print(df_model)

# best model: Random Forest
# lowest performace: Decision Trees: This model shows lower performance, especially in precision (84.44%). However, it has high recall (97.44%), meaning it's good at identifying positive cases but at the expense of generating more false positives.
# high precision, suitable for cases where false positives are costly.
