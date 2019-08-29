import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBoostClassifier
from sklearn.metrics import accuracy_score
data = pd.read_csv('Loan payments data.csv')
print(data.head())
X = data.iloc['Principal','terms','age','education','Gender']
Y=data['loan_status']
gender_label = LabelEncoder()
gender_label.fit(X.Gender)
X['Gender_Labelled']=gender_label.transform(X.Gender)
Ed_Label = LabelEncoder()
Ed_Label.fit(X.education)
X['Ed_Labelled'] = Ed_Label.transform(X.education)
Y_label = LabelEncoder
Y_label.fit(Y)
Y_labelled = Y_label.transform(Y)
X_Labelled = X.iloc('Principal','terms','age','Ed_Labelled','Gender_Labelled')
x_train,x_test,y_train,y_test = train_test_split(X_Labelled,Y_labelled)
model = XGBoostClassifier()
model.fit(x_train,y_train)
y_predict = model.predict(x_test)
print(accuracy_score(y_test,y_predict))