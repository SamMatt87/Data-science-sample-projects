import pandas as pd 
data = pd.read_csv('Heart.csv')
print(data.head())
from sklearn.model_selection import train_test_split
X = data.iloc[:,0:-1]
Y=data['target']
print(X.head())
print(Y.head())
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier
Neigh = KNeighborsClassifier(n_neighbors=5,metric = 'hamming')
Neigh.fit(xtrain,ytrain)
ypredict = Neigh.predict(xtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(ypredict,ytest))