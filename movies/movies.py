import pandas as pd 
import json
import matplotlib.pyplot as plt
import numpy as np
from yellowbrick.text import FreqDistVisualizer
from sklearn.cluster import KMeans
from anonymised import anonymiser
import anonymised
from vectorizer import vectorised

out, vectorizer = vectorised()
visualizer = FreqDistVisualizer(vectorizer.get_feature_names(),n=10)
visualizer.fit(out)
visualizer.poof()
model = KMeans(n_clusters = 10, init='k-means++', max_iter = 100, n_init=1)
model.fit(out)
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(0,10):
	print('Cluster %d:' % i)
	for ind in order_centroids[i][:20]:
 		print(' %s' % terms[ind])