import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn import tree
print('reading data')
train = pd.read_json("train.json")
ingredient_series =[]
print('spliting by dish')
for i in range (0,len(train)):
	print('dish ', str(i), ' of '+str(len(train)))
	for j in range(0,len(train['ingredients'][i])):
		ingredient_series.append([i,train['cuisine'][i],train['ingredients'][i][j]])
ingredients_list =pd.DataFrame(ingredient_series,columns=['dish no.','cuisine','ingredient'])
ingredients_list.to_csv('ingredients_list.csv')
print('group by cuisine')
ingredient_counts_series =[]
for i in (np.unique(ingredients_list['cuisine'])):
	cuisine_ingredients = ingredients_list[ingredients_list['cuisine'] == i]
	for j in (np.unique(cuisine_ingredients['ingredient'])):
		print('cuisine: ',i,' ingredient: ',j)
		ingredient_occurences = cuisine_ingredients[cuisine_ingredients['ingredient'] == j]
		ingredient_counts_series.append([i,j,ingredient_occurences['ingredient'].count()])
ingredient_counts = pd.DataFrame(ingredient_counts_series, columns=['cuisine','ingredient','count'])
ingredient_counts.to_csv('ingredient_counts.csv')
print('training tree')
X = ingredients_list['ingredient']
Y = ingredients_list['cuisine']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)
from sklearn.feature_extraction.text import CountVectorizer
CV = CountVectorizer()
x_counts = CV.fit_transform(X_train)
x_counts_df = pd.DataFrame(x_counts.toarray())
x_counts_df.columns = CV.get_feature_names()
print(x_counts_df)
Groot = tree.DecisionTreeClassifier(max_depth = 50)
Groot.fit(x_counts_df,Y_train)
Groot.score(x_counts_df,Y_train)
print('calculating error')
Groot_predict = Groot.predict(X_test)
Groot_Error = 0
for i in range(0,len(Y_test)):
	if Y_test[i]!=Groot_predict[i]:
		Groot_Error += 1
print('Errors:',Groot_Error)
Accuracy = ((len(Y_test)-Groot_Error)/len(Y_test))*100
print('Accuracy:',Accuracy)
