import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
import numpy as np
data = pd.read_csv('flight_delays_train.csv')
print(data.head())
data['month_tag'] = data['Month'].str.split('-').str[1]
data['day_tag'] = data['DayofMonth'].str.split('-').str[1]
data['weekday_tag'] = data['DayOfWeek'].str.split('-').str[1]
carr_enc = LabelEncoder()
#print(type(data['UniqueCarrier'].values.reshape(-1,1)))
carr_enc.fit(data['UniqueCarrier'])
data['carr_encoded'] = carr_enc.transform(data['UniqueCarrier'])
airp_enc = LabelEncoder()
airp_enc.fit(data['Origin']) 
data['Origin_encoded'] = airp_enc.transform(data['Origin'])
airp_enc2=LabelEncoder()
airp_enc2.fit(data['Dest']) 
data['Dest_encoded'] = airp_enc2.transform(data['Dest'])
YN_enc = LabelEncoder()
YN_enc.fit(data['dep_delayed_15min'])
data['Y_encoded'] = YN_enc.transform(data['dep_delayed_15min'])
print(data.head())
X = data[['month_tag','day_tag','weekday_tag','DepTime','carr_encoded','Origin_encoded','Distance','Dest_encoded']]
Y= data['Y_encoded']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(x_train,y_train)
y_predict = rf.predict(x_test)
from sklearn.metrics import accuracy_score
print(y_test)
print(y_predict)
print(accuracy_score(y_test,y_predict))
feature = pd.Series(rf.feature_importances_, index = X.columns).sort_values(ascending = False)
print(feature)
import matplotlib.pyplot as plt 
import seaborn as sb
sb.barplot(x=feature,y=feature.index)
plt.xlabel('Feature importance score')
plt.ylabel('Feature')
plt.title('Feature importance')
plt.legend()
plt.show()