import random
import operator
import copy
from random import seed
from grid import grid2 as grid
'''
1- INITIALIZE POPULATION
2- EVALUATE POPULATION
3- WHILE CONDITION NOT SATISFIED:
4-	REPRODUCE INDIVIDUALS
5-	   MUTATE INDIVIDUALS
6-       EVALUATE POPULATION
'''

Action 		= [1, 2, 3, 4]	#1= Turn Left, 2= Turn Right, 3= Move Forward, 4= Do Nothing
TotalPopulation = 20		#Population Size
TotalActions	= 28		#Total Actions in which max fitness is to be obtained
Population 	= []		#Will be filled by InitalizePopulation()
Pos		= [4, 5]
Direction	= 0		#0= North, 1= East, 2= South, 3= West
FitnessValue	= 0
FitIndividual	= []
Gen		= 1

def InitalizePopulation():
	for i in range(0, TotalPopulation):
		tmp = []
		for j in range(0, TotalActions):
			tmp.append(random.randint(1, 3))
		Population.append(tmp)
	return

def PerformStep(action, currPos, currDir, currGrid):
	x = currPos[0]
	y = currPos[1]
	d = currDir
	g = currGrid
	profit = 0

	'''
		SENSE/ACT LOGIC
	'''
	if action == 1: #Turn Left
		d -= 1
		if d == -1: d = 3
	elif action == 2: #Turn Right
		d = (d + 1) % 4
	elif action == 3: #Move Forward
		tmpCoords = [x, y]
		if d == 0: tmpCoords = [x-1,   y]
		if d == 1: tmpCoords = [x  , y+1]
		if d == 2: tmpCoords = [x+1,   y]
		if d == 3: tmpCoords = [x  , y-1]

		try:
			if g[tmpCoords[0]][tmpCoords[1]] != -1:
				profit += g[tmpCoords[0]][tmpCoords[1]]
				g[tmpCoords[0]][tmpCoords[1]] = 0
				x = tmpCoords[0]
				y = tmpCoords[1]
			else:
				pass
		except:
			pass
	else:
		pass
	
	
	return profit, [x, y], d, g

def EvaluatePopulation(printFlag):
	global Population
	global grid
	global Pos
	Evaluations = {} #{FitIndividual's index, FitnessValue}
	for i in range(0, len(Population)):
		currGrid = []
		for k in range(0, len(grid)):	#Making Deep Copy of grid in currGrid
			t = []
			for j in range(0, len(grid[k])):
				t.append(grid[k][j])
			currGrid.append(copy.deepcopy(t))
			
		currPos = [Pos[0], Pos[1]]
		currDir = 0
		Fitness = 0
		for step in Population[i]:
			profit, currPos, currDir, currGrid = PerformStep(step, currPos, currDir, currGrid)
			Fitness += profit
		global FitnessValue
		global FitIndividual
		if Fitness > FitnessValue:
			FitnessValue  = Fitness
			FitIndividual = copy.deepcopy(Population[i])
		Evaluations[i] = Fitness
		#print Population[i], Fitness
	if printFlag == 1:
		print "Fittest so far: ", FitIndividual, " ", FitnessValue
	return Evaluations

def ReproducePopulation(ind1, ind2):
	index = random.randint(0, len(ind1)-1)
	#print '\'\'', ind1, ' ', ind2, '\'\''
	for i in range(index, len(ind1)):
		ind1[i], ind2[i] = ind2[i], ind1[i]
	#print '\'\'', ind1, ' ', ind2, '\'\''
	return ind1, ind2

def MutatePopulation(ind1):
	index = random.randint(0, len(ind1)-1)
	m = random.randint(1, 3)
	ind1[index] = Action[m-1]
	return ind1

def ConditionSatisfied():
	if FitnessValue >= 20: 
		print 'Satisfied'
		return True
	else: return False

def SelectIndividual(res):
	#print 'res', res
	sorted_res = sorted(res.items(), key = operator.itemgetter(1))
	sorted_res = sorted_res[::-1]
	sample = []
	for i in sorted_res:	#(individual's index in Population, Value)
		for j in range(0, i[1]):
			sample.append(i[0])
	#print 'sample', sample
	random.shuffle(sample)
	return sample[random.randint(0, len(sample)-1)]
		
def GynaticAlgorithm():
	global Gen
	global Population
	InitalizePopulation()
	res = EvaluatePopulation(1)
	
	#Choosing Elite
	sorted_res = sorted(res.items(), key = operator.itemgetter(1))
	eliteInd   = sorted_res[len(sorted_res)-1][0]
	peasantInd = sorted_res[0][0]
	peasantInd2= sorted_res[1][0]
	while not ConditionSatisfied():
		Gen += 1
		print 'Generation', Gen
		#Remove peasant
		Population[peasantInd ] = copy.deepcopy(Population[eliteInd])
		Population[peasantInd2] = copy.deepcopy(Population[eliteInd])

		#resetting res as peasant is removed and elite is introduced
		#code can be optimized here, as I need to update only one value
		res = EvaluatePopulation(0)
		
		#Probabilistically selecting Next Generation's Parents
		newPopulation = []
		for i in range(0, TotalPopulation):
			newPopulation.append(copy.deepcopy(Population[SelectIndividual(res)]))
		Population = []
		Population = copy.deepcopy(newPopulation)
		
		#Parenst Making offsprings
		for i in xrange(0, TotalPopulation-1, 2):
			if random.randint(0, 10) > 2:
				Population[i], Population[i+1] = ReproducePopulation(Population[i], Population[i+1])
			if random.randint(0, 10) > 4:
				MutatePopulation(Population[i])
				MutatePopulation(Population[i+1])
		#Evaluations
		res        = EvaluatePopulation(1)
		eliteInd   = sorted_res[len(sorted_res)-1][0]
		peasantInd = sorted_res[0][0]
		peasantInd2= sorted_res[1][0]
		
	return

GynaticAlgorithm()
#InitalizePopulation()
#EvaluatePopulation()
