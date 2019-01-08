import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import numpy
import math
import graphviz
passengers = pd.read_csv('train.csv')
X=passengers.drop(['PassengerId','Survived','Name','Cabin','Ticket',],axis=1)
Y=passengers['Survived']
Label_Encoder = LabelEncoder()
Label_Encoder2 = LabelEncoder()
gender_encoded = Label_Encoder.fit_transform(X['Sex'])
port_encoded = Label_Encoder2.fit_transform(X['Embarked'])
X['Sex'] = gender_encoded
X['Embarked'] = port_encoded
X['Age']=X['Age'].fillna(X['Age'].mean())

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)
Groot = tree.DecisionTreeClassifier(max_depth = 50)
Groot.fit(X_train,Y_train)
Groot.score(X_train,Y_train)
print('Decision Tree')
Groot_predict = Groot.predict(X_test)
Y_test_list = list(Y_test)
Groot_Error = 0
for i in range(1,len(Y_test)):
	if Y_test_list[i]!=Groot_predict[i]:
		Groot_Error += 1
print('Errors:',Groot_Error)
Accuracy = ((len(Y_test)-Groot_Error)/len(Y_test))*100
print('Accuracy:',Accuracy)
groot_data = tree.export_graphviz(Groot, out_file ='out.dot', feature_names = X.columns, class_names = ['Survived','Dead'],filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(groot_data)
graph

print('Random Forest')
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100, max_features = 5, max_depth = 50)
forest = forest.fit(X_train,Y_train)
RF_predict = forest.predict(X_test)
RF_Error = 0
for i in range(1,len(Y_test)):
	if Y_test_list[i]!=RF_predict[i]:
		RF_Error += 1
print('Errors:',RF_Error)
RF_Accuracy = ((len(Y_test)-RF_Error)/len(Y_test))*100
print('Accuracy:',RF_Accuracy)

print('Linear Regression')
from sklearn.linear_model import LogisticRegression
Silverchair = LogisticRegression()
Silverchair.fit(X_train,Y_train)
Silverchair.score(X_train,Y_train)
print('Coefficient:\n',Silverchair.coef_)
print("Intercept:\n",Silverchair.intercept_)
Silverchair_predict = Silverchair.predict(X_test)
Silverchair_Error = 0
for i in range(1,len(Y_test)):
	if Y_test_list[i]!=Silverchair_predict[i]:
		Silverchair_Error += 1
print('Errors: ',Silverchair_Error)
Silverchair_accuracy = ((len(Y_test)-Silverchair_Error)/len(Y_test))*100
print('Accuracy: ',Silverchair_accuracy)

print('SVM')
from sklearn import svm
svm_model = svm.NuSVC(kernel = 'linear', nu = 0.4)
svm_model.fit(X_train,Y_train)
svm_model.score(X_train,Y_train)
svm_predict = svm_model.predict(X_test)
SVM_Error = 0
for i in range(1,len(Y_test)):
	if Y_test_list[i]!=svm_predict[i]:
		SVM_Error += 1
print('Errors:',SVM_Error)
SVM_Accuracy = ((len(Y_test)-SVM_Error)/len(Y_test))*100
print('Accuracy:',SVM_Accuracy)
#print(svm_model.decision_function(X))

print('Naive Bayes')
from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB(alpha = 0.4, fit_prior = False)
NB.fit(X_train,Y_train)
NB.score(X_train,Y_train)
NB_predict = NB.predict(X_test)
NB_Error = 0
for i in range(1,len(Y_test)):
	if Y_test_list[i]!=NB_predict[i]:
		NB_Error += 1
print('Errors:',NB_Error)
NB_Accuracy = ((len(Y_test)-NB_Error)/len(Y_test))*100
print('Accuracy:',NB_Accuracy)
#print(len(NB.predict_proba(X)))