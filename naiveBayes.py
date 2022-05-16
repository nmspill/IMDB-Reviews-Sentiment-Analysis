from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd


#data = pd.read_csv('data\\features.csv')
data = pd.read_csv("./data/features.csv")
features = ['positive adj count', 'negative adj count']
X = data[features] # Features
y = data['sentiment'] # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = MultinomialNB()
model.fit(X, y)


y_pred = model.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

print(cnf_matrix)
print('\n')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))