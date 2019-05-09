import numpy as np
import scipy.special
import math 
import time as t
#from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt

# for deleting extra spaces and [ ] while loading data from the file
def rectify(line):
    line = line.replace("[", "").replace("]", "").replace("  ", " ").rstrip()
    line = line.split(" ")

    val = ''
    while val in line:
        line.remove(val)

    return line
# read pixel values from the text files
def getData(fname):

    f = open(fname, 'r')
    content = f.readlines()
    f.close()

    Data = []
    temp = []

    stop = False
    for line in content:
        # reached end of one example (training or test)
        if(line.find("]") != -1):
            stop = True

        #copy elements to temp
        line = rectify(line)
        for j in line:
            temp.append(int(j))

        #stop copying now and append this temp to Data which is a 'list of lists'
        if(stop == True):
            Data.append(temp)
            temp = []
            stop = False

    #convert to array
    return np.asarray(Data)

#read target values from the text file
def getTarget(fname):

    target = []
    with open(fname) as f:
        for line in f:
            #temp = np.zeros((10), dtype=int)
            num = int(line.rstrip())
            #temp[num] = 1
            target.append(num)

    target = np.asarray(target)
    return target


def sigmoid(x):
    x = np.asfarray(x)
    return 1 / (1 + np.exp(-x))


mse_hist = []

class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
   
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)) #(30x784)
        #print('wih dimensions ' + str(self.wih.shape))

        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)) #(10x30)
        #print('who dimensions ' + str(self.who.shape))
      
        self.lr = learningrate
        self.activation_function = sigmoid

        pass

        #train the neural network
    def train(self, inputs_list, targets_list):

        inputs = np.array(inputs_list,ndmin=2).T #(784x1)
        #print('1. inputs dimensions : ' + str(inputs.shape))
        
        targets = np.array(targets_list,ndmin=2).T #(10x1)
        #print('2. targets dimensions : ' + str(targets.shape))

    
        hidden_inputs = np.dot(self.wih,inputs)  #(30x784).(784x1) = (30x1)
        hidden_outputs = self.activation_function(hidden_inputs)  #(30x1)
        #print('3. hidden outputs dimensions : ' + str(hidden_outputs.shape))


        final_inputs = np.dot(self.who,hidden_outputs) #(10x30).(30x1) = (10x1)
        final_outputs = self.activation_function(final_inputs) #(10x1)
        #print("4. final output dimensions : " + str(final_outputs.shape))
        
        #print("feedforward done!\n")

        # BACKPROPAGATION #
        # mse = mean_squared_error(targets,final_outputs)
        # mse_hist.append(mse)

        output_errors = targets - final_outputs #(10x1)
        #print('5. output errors dimensions :' + str(output_errors.shape))
 
        hidden_errors = np.dot(self.who.T, output_errors) #(30x10).(10x1) = (30x1)
        
        #print('6. hidden layer errors size : ' + str(hidden_errors) )
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr *np.dot((output_errors * final_outputs * (1.0 - final_outputs)), #(10x1).(1x30) = (10x30)
                   np.transpose(hidden_outputs)) 
        #print('7. updated output layer weight dimensions ' + str(self.who.shape))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * #(30x1).(1x784) = (30x784)
                    (1.0 - hidden_outputs)), np.transpose(inputs))
        #print('8. updated hidden layer weights dimensions' + str(self.wih.shape))
        return final_outputs

    #query the neural network
    def query(self, inputs_list):
        # convert input list to 2d array
        inputs = inputs_list.T #(784x1)

        # calcuclate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs) #(30x784).(784x1) = (30x784)
        # calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) #(30x1)

        # calculate signals  into final output layer
        final_inputs = np.dot(self.who, hidden_outputs) #(10x30).(30x1) = (10x1)
        # calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs) #(10x1)

        return final_outputs

# number of nodes
input_nodes = 784
hidden_nodes = 30
output_nodes = 10

# learning rate with 0.1
learning_rate = 0.01

# create an instance of neuralnetwork
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)



"""Load the data sets"""

print('Loading data...')

training_data = np.asarray(getData('train.txt'))
print training_data.shape, ' <='
train_labels = getTarget('train-labels.txt')
print train_labels.shape, ' <='
test_data = np.asarray(getData('test.txt'))
test_labels = getTarget('test-labels.txt') 




# train the neural network
epochs = 2
print('training...')

learning_rates = [0.01,0.05,0.1]


times = []
n.lr = learning_rates[0]
start = t.time()
for e in range(epochs):
    score = []
    for i in xrange(0, 60000):

        inputs = training_data[i]
        #inputs = ((inputs/255.0)*0.99) + 0.01
        inputs = ((inputs/255.0))

        #targets = np.zeros(output_nodes) + 0.01
        #targets[int(train_labels[i])] = 0.99

        targets = np.zeros(output_nodes) 
        targets[int(train_labels[i])] = 1

        label = np.argmax(n.train(inputs, targets))
        correct_label = train_labels[i]
        if(label == correct_label):
            score.append(1)
        else:
            score.append(0)
        pass
    
    score = np.asfarray(score)
    print('Epoch ' + str(e+1) + ' accuracy : ' + str(score.sum()/score.size))
    
    pass


end = t.time()
t1 = end-start
print('training took : ' + str(t1) + ' seconds\n')



# TESTING

# plt.plot(learning_rates,times)
# plt.xlabel('learning rates')
# plt.ylabel('time to train')
# plt.title('Time to train vs Learning rate')
# print(times)
# plt.show()

scorecard = []

print('testing...')
start = t.time()
for i in range(10000):
    # correct answer is first value
    correct_label = test_labels[i]
    # print(correct_label, "correct label")

    outputs = n.query(test_data[i])

    # the index of the highest value corresponds to the label
    label = np.argmax(outputs)
    # print(label, "network's answer\n")

    if (label == correct_label):
        scorecard.append(1)

    else:
        scorecard.append(0)

    pass

    
end = t.time()
print('testing took : ' + str(end-start) + ' seconds')

scorecard_array = np.asfarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)
print('scorecard size ' + str(scorecard_array.size))
