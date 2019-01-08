import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score

data = pd.read_csv('weatherAUS.csv')
data=data.replace('No',0)
data=data.replace('Yes',1)
Month = pd.DatetimeIndex(data.Date).month.values
data['Month']=Month
print(data.Month)
direction = ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW']
direction_numeric=[0,22.5,45,67.5,90,112.5,135,157.5,180,202.5,225,247.5,270,292.5,315,337.5]
for i in range(0,len(direction)):
	data = data.replace(direction[i],direction_numeric[i])
data = data.fillna(0)
print(data.head())
x=data[['MinTemp','MaxTemp','Rainfall','Evaporation', 'Sunshine','WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am','WindSpeed3pm','Humidity9am','Humidity3pm','Pressure9am','Pressure3pm','Cloud9am','Cloud3pm','Temp9am','Temp3pm','RainToday','Month']]
y=data['RainTomorrow']
print(x.head())
print(y.head())

x_train, x_test, y_train, y_test = train_test_split(x,y)

clf = tree.DecisionTreeClassifier(max_depth = 20, min_samples_leaf = 1000)
clf = clf.fit(x_train,y_train)
y_predict=clf.predict(x_test)
score = accuracy_score(y_test,y_predict)
print(score)

dot_data = tree.export_graphviz(clf,
                                feature_names=x.columns,
                                out_file=None,
                                filled=True,
                                rounded=True)

import pydotplus
graph = pydotplus.graph_from_dot_data(dot_data)
import collections
colors = ('turquoise', 'orange')
edges = collections.defaultdict(list)
for edge in graph.get_edge_list():
    edges[edge.get_source()].append(int(edge.get_destination()))
for edge in edges:
    edges[edge].sort()    
    for i in range(2):
        dest = graph.get_node(str(edges[edge][i]))[0]
        dest.set_fillcolor(colors[i])
graph.write_png('tree.png')