import sys
import numpy as np
from utilities import readInput, readLabels, writeWeights
from train import train
from test import test

#testing in test.py file
#training in train.py file

import warnings
if not sys.warnoptions: warnings.simplefilter("ignore")



if(sys.argv[1] == 'test'):
	testingData   = readInput(sys.argv[2])
	testingLabels = readLabels(sys.argv[3])
	test(testingData, testingLabels)



if(sys.argv[1] == 'train'):
	trainingData   = readInput(sys.argv[2])
	trainingLabels = readLabels(sys.argv[3])
	w1, w2 = train(trainingData, trainingLabels)
	writeWeights(w1, w2)

