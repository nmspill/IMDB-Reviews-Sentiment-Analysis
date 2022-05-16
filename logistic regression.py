from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

data = pd.read_csv('data\\features.csv')
features = ['positive adj count','negative adj count','word count','contains !',"contains 'no'"]
X = data[features] # Features
y = data['sentiment'] # Target variable
X = X.to_numpy()
y = y.to_numpy()

y_pred_lst = []
y_test_lst = []

logisticRegr = LogisticRegression()

def scoring_metrics(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],'fn': cm[1, 0], 'tp': cm[1, 1], 'accuracy': metrics.accuracy_score(y, y_pred), 'precision': metrics.precision_score(y, y_pred),'recall': metrics.recall_score(y, y_pred)}

cv_results = cross_validate(logisticRegr, X, y, cv=10,scoring=scoring_metrics)
cm = np.array([[cv_results['test_tp'].mean(), cv_results['test_fp'].mean()], [cv_results['test_fn'].mean(), cv_results['test_tn'].mean()]])
cm_displayed = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])


print('Accuracy:', cv_results['test_accuracy'].mean())
print('Precision:', cv_results['test_precision'].mean())
print('Recall:', cv_results['test_recall'].mean())

cm_displayed.plot(values_format='.1f')
plt.show()










