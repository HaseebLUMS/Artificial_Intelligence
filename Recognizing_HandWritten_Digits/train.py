import sys
import numpy as np
from utilities import sigmoid as activationFunction
import warnings
if not sys.warnoptions: warnings.simplefilter("ignore")

import time
'''
[60k, 784]    [30, 1]   	[10, 1]
	|         					|
    |             |				|
    |	  	      |				|
    |		      |				|
    |							|
	|							|
'''
def train(trainingData, trainingLabels):
	start_time = time.time()
	#Initializing random weights with mean 0 and sd 1
	layer1Weights = np.random.normal(0.0, pow(30, -0.5), (30, 784))
	layer2Weights = np.random.normal(0.0, pow(10, -0.5), (10,  30))

	trials = 2
	while(trials):
		LEN = len(trainingData)
		for i in range(0, LEN):
			trainingData[i]=((trainingData[i]-np.mean(trainingData[i]))/np.std(trainingData[i])) #Normalizing
			

			l1Output  = np.array(np.dot(layer1Weights, trainingData[i].T)) #(30, 784) * (784, 1)
			l1Output  = (np.vectorize(activationFunction))(l1Output)


			l2Output  = np.dot(layer2Weights, l1Output) #(10, 30) * (30, 1)
			l2Output  = (np.vectorize(activationFunction))(l2Output)


			o         = l2Output
			y         = np.zeros(10)
			y[trainingLabels[i]] = 1


			#Back Propagation
			err_l2O = np.array(y - o)
			err_l1O = np.dot(layer2Weights.T, err_l2O) #(30, 10) * (10 * 1)


			sec = np.array([(err_l2O *o* (1 -o))])
			learningRate = float((sys.argv[4]))
			

			#Updating Weights
			layer2Weights += learningRate*np.dot(np.array([(err_l2O *o* (1 -o))]).T,             np.array([l1Output])) #(10, 1) *(1, 30)
			layer1Weights += learningRate*np.dot(np.array([(err_l1O * l1Output * (1 - l1Output))]).T, np.array([trainingData[i]]))

		print (layer1Weights.shape, layer2Weights.shape)
		trials  -= 1
	print("--- %s seconds ---" % (time.time() - start_time))

	return layer1Weights, layer2Weights