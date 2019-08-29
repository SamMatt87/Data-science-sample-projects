from yellowbrick.text import FreqDistVisualizer
def visualised(a,b):
	visualizer = FreqDistVisualizer(b.get_feature_names(),n=10)
	visualizer.fit(a)
	visualizer.poof()