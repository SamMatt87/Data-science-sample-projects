from scipy.io import loadmat
import numpy
mnist = loadmat('mnist-original.loadmat')
mnist_data = mnist['data'].T
mnist_label = mnist['label'][0]
print (mnist)
print(mnist_data)
print(mnist_label)