import pandas as pd
import numpy as np 
import json
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
print(stop_words)
Credits =pd.read_csv('tmdb_5000_credits.csv')
Movies = pd.read_csv('tmdb_5000_movies.csv')
global Movies_count 
Movies_count = len(Movies)
def anonymiser(a):
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