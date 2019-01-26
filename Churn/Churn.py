from sklearn.cluster import KMeans
import pandas as pd 
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(data.head())
X=data.iloc[:,1:20]
Y=data['Churn']
print(X.head())
print(Y)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)
print(x_train)
print(y_train)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(x_train,y_train)
kmeans.labels_
y_predict=kmeans.preict(test_x)
kmeans.cluster_centers_
