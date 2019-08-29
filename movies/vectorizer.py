from anonymised import anonymiser
import anonymised
from sklearn.feature_extraction.text import CountVectorizer
def vectorised():
	words=[]
	#print(words)
	for a in range(0,anonymised.Movies_count):
		words.append(anonymiser(a))
		if (a+1) % 1000 == 0: 
			print((a+1),' plots read')
	vectorizer=CountVectorizer()
	out = vectorizer.fit_transform(words)
	return out, vectorizer