import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

data = pd.read_csv('train.csv')
print(data.head())
#print(data['tags'][1][1:-1].split(',')[1].split("'")[1])
code = []
question = []
for i in range(0,len(data['tags'])):
	for j in range (0,len(data['tags'][i][1:-1].split(','))):
		x=data['tags'][i][1:-1].split(',')[j].split("'")[1]
		if x.lower() == 'r':
			code.append('R')
			question.append(data['title'][i])
		elif x.lower() == 'python':
			code.append('Python')
			question.append(data['title'][i])
		elif x.lower() == 'java':
			code.append('Java')
			question.append(data['title'][i])
		elif x.lower() == 'mysql':
			code.append('sql')
			question.append(data['title'][i])
		else:
			code.append('N/A')
			question.append(data['title'][i])
#print(code)
#print(question)
tagged = pd.DataFrame({'Question':question,'Code':code})
print(tagged.head())

def to_words(raw_text):
	text = BeautifulSoup(raw_text, features = "html.parser").get_text()
	letters = re.sub("[^a-zA-Z]"," ",text)
	lower_case = letters.lower()
	words = lower_case.split()
	stops = set(stopwords.words('english'))
	meaningful_words = [w for w in words if not w in stops]
	return (",".join(meaningful_words))

tagged_reduced = tagged[tagged.Code != 'N/A']
print(tagged_reduced.head())

x_train, x_test, y_train, y_test = train_test_split(tagged_reduced.Question,tagged_reduced.Code,test_size=0.2)
y_train = np.asarray(y_train)
num_qs = x_train.size 
print(x_train.values[1])
clean_q_train =[]
for i in range (0,num_qs):
	clean_q_train.append(to_words(x_train.values[i]))
	if i%1000 == 0:
		print("Question %d of %d\n"%(i+1,num_qs))
	elif i==(num_qs-1):
		print('finished')
vectorizer = CountVectorizer(analyzer='word', tokenizer = None, stop_words=None, max_features = 5000)
train_data_features = vectorizer.fit_transform(clean_q_train)
#train_data_features = train_data_features.values()
vocab = vectorizer.get_feature_names()
dist=np.sum(train_data_features, axis=0)
print(dist)
for i in range(0,len(vocab)):
	print(vocab[i],dist[0,i])
#print (train_data_features.shape)
#print(train_data_features)
#for i in range(0,num_qs):
	#print (train_data_features[i])
clean_q_test=[]
num_test_qs = x_test.size
for i in range (0,num_test_qs):
	clean_q_test.append(to_words(x_test.values[i]))
	if i%1000 == 0:
		print("Question %d of %d\n"%(i+1,num_test_qs))
	elif i==(num_test_qs-1):
		print('finished')
test_data_features = vectorizer.transform(clean_q_test)
#train_data_features = np.asarray(test_data_features)
#print(train_data_features.dtype)
clf = SVC(gamma='auto')
clf.fit(train_data_features,y_train)
y_predict = clf.predict(test_data_features)
score = accuracy_score(y_test,y_predict)
print(score)
