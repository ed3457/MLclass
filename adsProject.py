import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

df_net = pd.read_csv('Social_Network_Ads.csv')
print(df_net.head())
df_net.drop(columns = ['User ID'], inplace=True)
print(df_net.head())

print(df_net.describe())

sns.displot(df_net['EstimatedSalary'])
plt.show()

le = LabelEncoder()
df_net['Gender']= le.fit_transform(df_net['Gender'])

df_net.corr()
sns.heatmap(df_net.corr())

plt.show()

df_net.drop(columns=['Gender'], inplace=True)

X = df_net.iloc[:, :-1].values
y = df_net.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Decision Tree Classification
classifier = DecisionTreeClassifier(max_depth=4,criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Prediction
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

print(accuracy_score(y_test, y_pred))
print(f'Classification Report: \n{classification_report(y_test, y_pred)}')

print(f"F1 Score : {f1_score(y_test, y_pred)}")

cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.show()