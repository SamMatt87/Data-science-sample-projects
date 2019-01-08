import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LogisticRegression
data = pd.read_csv('creditcardcsvpresent.csv')
print(data.head())
print(data.Merchant_id.nunique())

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, log_loss
data=data.replace('N',0)
data=data.replace('Y',1)
X= data.iloc[:,2:11]
Y = data[['isFradulent']]

print(X)
print(Y.head())
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2)

logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)
y_pred_proba = logreg.predict_proba(x_test)[:,1]
[fpr,tpr,thr] = roc_curve(y_test,y_pred_proba)
print('Train/Test split results:')
print(logreg.__class__.__name__+" accuracy is %2.3f" % accuracy_score(y_test, y_pred))
print(logreg.__class__.__name__+" log_loss is %2.3f" % log_loss(y_test, y_pred_proba))
print(logreg.__class__.__name__+" auc is %2.3f" % auc(fpr, tpr))
idx = np.min(np.where(tpr > 0.95))
plt.figure()
plt.plot(fpr,tpr, color='coral', label = 'ROC Curve (area=%0.3f)'%auc(fpr,tpr))
plt.plot([0,1],[0,1],'k--')
plt.plot([0,fpr[idx]],[tpr[idx],tpr[idx]],'k--',color='blue')
plt.plot([fpr[idx],fpr[idx]],[0,tpr[idx]],'k--',color='blue')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate (1-specificity)', fontsize=14)
plt.ylabel('True positive rate (recall)', fontsize=14)
plt.legend(loc='lower right')
plt.show()

print("Using a threshold of %.3f " % thr[idx] + "guarantees a sensitivity of %.3f " % tpr[idx] +  
      "and a specificity of %.3f" % (1-fpr[idx]) + 
      ", i.e. a false positive rate of %.2f%%." % (np.array(fpr[idx])*100))