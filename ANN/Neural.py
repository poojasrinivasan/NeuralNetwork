import pandas as pd
import random
import math
import numpy as np
import sys
import logging
LOG_FILENAME = 'example.log'
logging.basicConfig(filename=LOG_FILENAME,level=logging.DEBUG)

# network initialization
def networkInitialization(numberOfInputs, numberOfOutputs, numberOfHiddenLayers, numberOfNeuronsPerLayer):
    weightList = list()
    for index in range(numberOfHiddenLayers + 1):
        if index == 0:
            hiddenLayer = [[random.random() for i in range(numberOfInputs + 1)] for i in
                           range(numberOfNeuronsPerLayer[index])]
        elif index == numberOfHiddenLayers:
            hiddenLayer = [[random.random() for i in range(numberOfNeuronsPerLayer[index - 1] + 1)] for i in
                           range(numberOfOutputs)]
        else:
            hiddenLayer = [[random.random() for i in range(numberOfNeuronsPerLayer[index - 1] + 1)] for i in
                           range(numberOfNeuronsPerLayer[index])]
        weightList.append(hiddenLayer)
    return weightList


# activation
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return transfer(activation)


# Sigmoid Function
def transfer(activation):
    return 1.0 / (1.0 + math.exp(-activation))


# Forward Propagation
def forwardPropagate(weightList, dataRow):
    inputs = dataRow[:-1]
    eachNeuronOutputs = list()
    eachNeuronOutputs.append(dataRow[:-1])
    for eachLayerIndex in range(len(weightList)):
        newInputs = []
        for eachNeuronIndex in range(len(weightList[eachLayerIndex])):
            activation = activate(weightList[eachLayerIndex][eachNeuronIndex], inputs)
            newInputs.append(activation)
        inputs = newInputs
        eachNeuronOutputs.append(newInputs)
    return eachNeuronOutputs


# BackWard Propagate
def backwardPropagate(outputs, expectedOutput, weightList, learningRate):
    # expectedOutput = [0,1]
    deltaList = []
    for indexOutputLayer in reversed(range(len(outputs))):
        newDelta = list()
        if indexOutputLayer == len(outputs) - 1:
            for index in range(len(outputs[indexOutputLayer])):
                delta = outputs[indexOutputLayer][index] * (1 - outputs[indexOutputLayer][index]) * (
                    expectedOutput[index] - outputs[indexOutputLayer][index])
                newDelta.append(delta)
            deltaList = newDelta
        elif indexOutputLayer == 0:
            for index in range(len(outputs[indexOutputLayer])):
                for indexWeight in range(len(weightList[indexOutputLayer])):
                    weightList[indexOutputLayer][indexWeight][index] += learningRate * deltaList[indexWeight] * \
                                                                       outputs[indexOutputLayer][index]
            for indexWeight in range(len(weightList[indexOutputLayer])):
                weightList[indexOutputLayer][indexWeight][-1] += learningRate * deltaList[indexWeight]
        else:
            for index in range(len(outputs[indexOutputLayer])):
                sum = 0
                for indexWeight in range(len(weightList[indexOutputLayer])):
                    sum += weightList[indexOutputLayer][indexWeight][index] * deltaList[indexWeight]
                    weightList[indexOutputLayer][indexWeight][index] += learningRate * deltaList[indexWeight] * \
                                                                        outputs[indexOutputLayer][index]
                delta = outputs[indexOutputLayer][index] * (1 - outputs[indexOutputLayer][index]) * sum
                newDelta.append(delta)
            for indexWeight in range(len(weightList[indexOutputLayer])):
                weightList[indexOutputLayer][indexWeight][-1] += learningRate * deltaList[indexWeight]
            deltaList = newDelta


def train_network(weightList, traininigDataSet, learningRate, noOfIteration, numberOfOutputs):
    for iter in range(noOfIteration):
        sum_error = 0
        for row in traininigDataSet:
            outputs = forwardPropagate(weightList, row)
            expected = [0 for i in range(numberOfOutputs)]
            expected[int(row[-1]) - 1] = 1
            #actuals = maxOutput(outputs[len(outputs) - 1])
            # sum_error += (sum([(expected[i] - actuals[i]) ** 2 for i in range(len(expected))]) / 2)
            sum_error += sum([(expected[i] - outputs[len(outputs) - 1][i]) ** 2 for i in range(len(expected))])
            backwardPropagate(outputs,expected,weightList,learningRate)
        sum_error = sum_error/len(traininigDataSet)
        print('Iteration=%d, Error=%.8f' % (iter+1,  sum_error))
        precisedError = '%.8f' % sum_error
        if float(precisedError) == 0.0:
            break

def printWeights(weights):
    for layer in range(len(weights)):
        print("Layer " + str(layer) + ":")
        for col in range(len(weights[layer][0])):
            neuronWeights = []
            for row in range(len(weights[layer])):
                neuronWeights.append(weights[layer][row][col])
            if(col == len(weights[layer][0])-1):
                print("\t Bias Term :" + str(neuronWeights))
            else:
                print("\t Neuron " + str(col+1)+ " : " + str(neuronWeights))

def maxOutput(Lastoutputs):
    actuals = [0 for i in range(len(Lastoutputs))]
    index = Lastoutputs.index(max(Lastoutputs))
    actuals[index] = 1;
    return actuals

def testTheModel(dataSet, weightList,numberOfOutputs):
    sumError = 0
    count = 0
    for data in dataSet:
        outputs = forwardPropagate(weightList,data)
        expected = [0 for i in range(numberOfOutputs)]
        expected[int(data[-1]) - 1] = 1
        sumError += sum([(expected[i] - outputs[len(outputs) - 1][i]) ** 2 for i in range(len(expected))])
        actuals = maxOutput(outputs[len(outputs) - 1])
        if actuals == expected:
            count = count + 1
    sumError = sumError / len(dataSet)
    accuracy = count/len(dataSet)
    outs = []
    outs.append(sumError)
    outs.append(accuracy)
    return outs





# Main Code
userInput = sys.argv[1].split(" ")
preprosessedPath = userInput.pop(0)
splitRatio = int(userInput.pop(0))/100
numberofIteration = int(userInput.pop(0))
numberOfHiddenLayers = int(userInput.pop(0))
numberOfNeuronsPerLayer = [ int(x) for x in userInput]
df = pd.read_csv(preprosessedPath, header=None)
dataset = df.values



numberOfInputs = len(dataset[0]) - 1
numberOfOutputs = len(set([row[-1] for row in dataset]))


#split
msk = np.random.rand(len(df)) < splitRatio
train = df[msk].values
test = df[~msk].values
weightList = networkInitialization(numberOfInputs, numberOfOutputs, numberOfHiddenLayers, numberOfNeuronsPerLayer)
#printWeights(weightList)
# print(weightList)
# o = forwardPropagate(weightList, dataset[0])
# backwardPropagate(o, [0, 1], weightList, 0.5)
# print(weightList)
learningRate = 0.8
train_network(weightList,train,learningRate,numberofIteration,numberOfOutputs)
printWeights(weightList)
outsTraining = testTheModel(train,weightList,numberOfOutputs)
print("Total Training Error : " + str(outsTraining[0]))
print("Accuracy for Training Data: " + str(outsTraining[1]*100)+"%")
outsTesting = testTheModel(test,weightList,numberOfOutputs)
print("Total Testing Error : " + str(outsTesting[0]))
print("Accuracy for Testing Data: " + str(outsTesting[1]*100)+"%")
logging.info('Split Percent : ' + str(splitRatio*100)+'%' + ' , Number of Iteration : ' + str(numberofIteration) + " , Number of Hidden Layers : " + str(numberOfHiddenLayers) + ""
                " , Number of Neurons per Hidden Layer : " + str(numberOfNeuronsPerLayer) + " , Learning Rate : " +
             str(learningRate) +" , Training Error : "
              + str(outsTraining[0]) + " , Testing Error : " + str(outsTesting[0]))