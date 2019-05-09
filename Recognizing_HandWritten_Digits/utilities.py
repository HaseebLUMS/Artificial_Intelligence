import sys
import numpy as np
import warnings
if not sys.warnoptions: warnings.simplefilter("ignore")



def readInput(testFile):
	file = open(testFile, 'r')
	rows = file.readlines()
	file.close()

	end = 0
	partialrow   = []
	completeData = []
	for r in rows:
		if r.find(']') != -1 : end = 1
		r = r.replace(']', '').replace('[', '').replace('  ', ' ').rstrip()
		r = r.split(' ')
		r = filter(lambda a: a != '', r)
		int_r = []
		for ele in r: int_r.append(int(ele))
		partialrow.extend(int_r)
		if end: 
			completeData.append(partialrow)
			partialrow = []
			end = 0

	return np.array(completeData)




def readLabels(labelFile):
	labels = []
	file = open(labelFile, 'r')
	rows = file.readlines()
	file.close()
	for l in rows: labels.append(int(l.rstrip()))
	return np.array(labels)




def sigmoid(x):
	x = np.array(x)
	return 1 / (1 + np.exp(-x))




def readWeights(name):
	file = open(name)
	bigStr = file.read()
	lis = bigStr.split(' ')
	w1 = np.empty((30, 784), dtype=float)
	k = 0
	for i in range(0, 30):
		for j in range(0, 784):
			w1[i][j] = float(lis[k])
			k += 1
	w2 = np.empty((10, 30), dtype=float)
	for i in range(0, 10):
		for j in range(0, 30):
			w2[i][j] = float(lis[k])
			k += 1
	return w1, w2




			
def writeWeights(w1, w2):
	sw1 = 30*784
	sw2 = 10*30
	open('netWeights.txt', 'w').close() #for clearing the file
	f = open('netWeights.txt', 'a')
	for i in range(0, 30):
		for j in range(0, 784):
			f.write(str(w1[i][j]))
			f.write(' ')

	for i in range(0, 10):
		for j in range(0, 30):
			f.write(str(w2[i][j]))
			f.write(' ')
	f.close()

