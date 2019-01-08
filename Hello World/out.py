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
submission = pd.read_csv('test.csv')
submission = submission.drop(['PassengerId','Name','Cabin','Ticket',],axis=1)
Label_Encoder = LabelEncoder()
Label_Encoder2 = LabelEncoder()
gender_encoded = Label_Encoder.fit_transform(X['Sex'])
port_encoded = Label_Encoder2.fit_transform(X['Embarked'])
sud_gender_encoded = Label_Encoder.fit_transform(X['Sex'])
sub_port_encoded = Label_Encoder2.fit_transform(X['Embarked'])
X['Sex'] = gender_encoded
X['Embarked'] = port_encoded
X['Age']=X['Age'].fillna(X['Age'].mean())
submission['Sex']= sud_gender_encoded
submission['Embarked'] = sub_port_encoded
submission['Age'] = submission['Age'].fillna(submission['Age'].mean())
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)
Y_test_list = list(Y_test)
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
submission_predict = forest.predict(submission)
out = pd.concat([submission['PassengerId'],submission_predict], axis=1)
print(out)