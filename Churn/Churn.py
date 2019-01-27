from sklearn.cluster import KMeans
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
#print(data.head())
X=data.iloc[:,1:20]
Y=data['Churn']
y_label = LabelEncoder()
y_label.fit(Y)
X['Partner_labeled'] = y_label.transform(X['Partner'])
X['Dependents_labeled'] = y_label.transform(X['Dependents'])
X['PhoneService_labeled'] = y_label.transform(X['PhoneService'])
X['PaperlessBilling_labeled'] = y_label.transform(X['PaperlessBilling'])
y_labeled = y_label.transform(Y)

lines_label = LabelEncoder()
lines_label.fit(X['MultipleLines'])
X['MultipleLines_label'] = lines_label.transform(X['MultipleLines'])

Internet_type = LabelEncoder()
Internet_type.fit(X['InternetService'])
X['InternetService_label'] = Internet_type.transform(X['InternetService'])

Int_ext_label = LabelEncoder()
Int_ext_label.fit(X['OnlineSecurity'])
X['OnlineSecurity_label'] = Int_ext_label.transform(X['OnlineSecurity'])
X['OnlineBackup_label'] = Int_ext_label.transform(X['OnlineBackup'])
X['DeviceProtection_label'] = Int_ext_label.transform(X['DeviceProtection'])
X['TechSupport_label'] = Int_ext_label.transform(X['TechSupport'])
X['StreamingTV_label'] = Int_ext_label.transform(X['StreamingTV'])
X['StreamingMovies_label'] = Int_ext_label.transform(X['StreamingMovies'])

cont_label = LabelEncoder()
cont_label.fit(X['Contract'])
X['Contract_label'] = cont_label.transform(X['Contract'])

pay_label = LabelEncoder()
pay_label.fit(X['PaymentMethod'])
X['PaymentMethod_label'] = pay_label.transform(X['PaymentMethod'])

gender_label = LabelEncoder()
gender_label.fit(X['gender'])
X['Gender_label'] = gender_label.transform(X['gender'])

#x_labeled = 
x_labeled=X[['Gender_label','SeniorCitizen','Partner_labeled','Dependents_labeled',
	'tenure','PhoneService_labeled','MultipleLines_label','InternetService_label',
	'OnlineSecurity_label','OnlineBackup_label','DeviceProtection_label','TechSupport_label',
	'StreamingTV_label','StreamingMovies_label','Contract_label','PaperlessBilling_labeled',
	'PaymentMethod_label','MonthlyCharges','TotalCharges']]
print(X.head())
#print(X.head())
#print(y_labeled)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_labeled,y_labeled,test_size=0.2)
#print(x_train)
#print(y_train)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(x_train,y_train)
kmeans.labels_
y_predict=kmeans.preict(test_x)
kmeans.cluster_centers_
