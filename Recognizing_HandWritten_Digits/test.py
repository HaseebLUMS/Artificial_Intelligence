import sys
import numpy as np
from utilities import sigmoid as activationFunction
from utilities import readWeights
import warnings

if not sys.warnoptions: warnings.simplefilter("ignore")




'''
[60k, 784]    [30, 1]   	[10, 1]
	|         					|
    |             |				|
    |	  	      |				|
    |		      |				|
    |							|
	|							|
'''
def test(testingData, testingLabels):

	w1, w2 = readWeights(sys.argv[4])
	s = 0
	n = 0

	LEN = len(testingData)
	for i in range(0, LEN):
		# testingData[i]=((testingData[i]-np.mean(testingData[i]))/np.std(testingData[i])) #Normalizing
		
		l1Output  = np.array(np.dot(w1, testingData[i].T)) #(30, 784) * (784, 1)
		l1Output  = (np.vectorize(activationFunction))(l1Output)

		l2Output  = np.dot(w2, l1Output) #(10, 30) * (30, 1)
		l2Output  = (np.vectorize(activationFunction))(l2Output)

		o         = np.argmax(l2Output)
		y         = testingLabels[i]

		if o == y:
			s += 1
		n += 1

	s += 0.0
	acc = float(s/n) * 100
	print "Epoch # 2 =====> ", s , "/", n, " files correctly Classified"
	print "Accuracy = ", acc, '%', "-----------   Error", 100 - acc, '%'


