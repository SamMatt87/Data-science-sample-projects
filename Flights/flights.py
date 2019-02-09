import pandas as pd 
from sklearn.preprocessing import OneHotEncoder
data = pd.read_csv('flight_delays_train.csv')
print(data.head())
data['month_tag'] = data['Month'].str.split('-').str[1]
data['day_tag'] = data['DayofMonth'].str.split('-').str[1]
data['weekday_tag'] = data['DayOfWeek'].str.split('-').str[1]
carr_enc = OneHotEncoder()
print(type(data['UniqueCarrier'].values.reshape(-1,1)))
carr_enc.fit(data['UniqueCarrier'].values.reshape(-1,1))
data['carr_encoded'] = carr_enc.transform(data['UniqueCarrier'].values.reshape(-1,1))
airp_enc = OneHotEncoder()
airp_enc.fit(data['Origin'].values.reshape(-1,1)) 
data['Origin_encoded'] = airp_enc.transform(data['Origin'].values.reshape(-1,1))
#airp_enc.fit(data['Dest'].values.reshape(-1,1)) 
#data['Dest_encoded'] = airp_enc.transform(data['Dest'].values)
print(data.carr_encoded)