import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn import tree
#print('reading data')
train = pd.read_json("train.json")
print('vectorising')

Y = train['cuisine']
#print(train.head())
train['text'] = str(train['ingredients'])[1:-1]
print(train.head())
X = train['text']
#print(X)
#X=X.apply(lambda x: ','.join(x))
#X=X.apply(lambda x: x.split(','))
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)
print(X_train)
from sklearn.feature_extraction.text import CountVectorizer
CV = CountVectorizer()
x_counts = X_train.apply(lambda x : CV.fit_transform(x))
x_counts_df = pd.DataFrame(x_counts)
#print(x_counts_df.head())
x_counts_df.columns = CV.get_feature_names()
# print('training tree')
# Groot = tree.DecisionTreeClassifier(max_depth = 50)
# Groot.fit(x_counts_df,Y_train)
# Groot.score(x_counts_df,Y_train)
# print('calculating error')
# Groot_predict = Groot.predict(X_test)
# Groot_Error = 0
# for i in range(0,len(Y_test)):
# 	if Y_test[i]!=Groot_predict[i]:
# 		Groot_Error += 1
# print('Errors:',Groot_Error)
# Accuracy = ((len(Y_test)-Groot_Error)/len(Y_test))*100
# print('Accuracy:',Accuracy)