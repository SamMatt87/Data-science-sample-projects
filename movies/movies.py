import pandas as pd 
import json
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
from yellowbrick.text import FreqDistVisualizer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
stop_words = set(stopwords.words('english'))
print(stop_words)
Credits =pd.read_csv('tmdb_5000_credits.csv')
Movies = pd.read_csv('tmdb_5000_movies.csv')
#print(json.loads(Credits['cast'][1])[1]["character"])
#print(Movies['overview'][1])
def anonymised(a):
	word = str(Movies['overview'][a]).split(' ')
	for i in json.loads(Credits['cast'][a]):
		if i['order']<=10:
			name = str(i['character']).split(' ')
			for j in name:
				for k in range (0,len(word)):
					if word[k][0:len(j)]==j:
				 		if len(word[k])==len(j):
				 			word[k] ='character_name'
				 		else:
				 			word[k]='character_name'+word[k][(len(j)+1):]
	l=0
	while (l+1) <len(word):
		if word[l] in stop_words:
			del word[l]
		elif word[l] == 'character_name' and word[l+1][0:15]=='character_name':
			del word[l]
		else:
			l=l+1
	return(' '.join(word))
words=[]
#print(words)
for a in range(0,len(Movies)):
	words.append(anonymised(a))
	if (a+1) % 1000 == 0: 
		print((a+1),' plots read')
vectorizer=CountVectorizer()
out = vectorizer.fit_transform(words)
#print(vectorizer.get_feature_names())
#print(out.toarray())
#print(out.toarray().sum(axis=0))
# ypos = np.arange(len(out.toarray().sum(axis=0).tolist()))
# plt.bar(ypos,height=[out.toarray().sum(axis=0).tolist()])
# plt.xticks(vectorizer.get_feature_names())
visualizer = FreqDistVisualizer(vectorizer.get_feature_names(),n=10)
visualizer.fit(out)
visualizer.poof()
model = KMeans(n_clusters = 5, init='k-means++', max_iter = 100, n_init=1)
model.fit(out)
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(0,10):
	print('Cluster %d:' % i)
	for ind in order_centroids[i][:20]:
 		print(' %s' % terms[ind])