from numpy import exp, array, random, dot
import sys


# Nonliniar Sigmoid Function
# Can return derivative of sigmoid if 'der' set to true
def sigmoid(x, der=False):
    # Returns derivative of sigmoid at position x
    if (der == True):
        return x * (1 - x)

    # Returns sigmoid at position x
    return 1 / (1 + exp(-x))

# Input training example    
input_train = array([[0, 0, 0, 0, 0],
           [0, 0, 0, 1, 1],
           [0, 0, 1, 0, 1],
           [0, 0, 1, 1, 0],
           [0, 1, 0, 0, 1],
           [0, 1, 0, 1, 0],
           [0, 1, 1, 0, 0],
           [0, 1, 1, 1, 1],
           [1, 0, 0, 0, 1],
           [1, 0, 0, 1, 0],
           [1, 0, 1, 0, 0],
           [1, 0, 1, 1, 1],
           [1, 1, 0, 0, 0],
           [1, 1, 0, 1, 1],
           [1, 1, 1, 0, 1],
           [1, 1, 1, 1, 0],
           [0, 0, 0, 0, 1],
           [0, 0, 0, 1, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 1, 1],
           [0, 1, 0, 0, 0],
           [0, 1, 0, 1, 1],
           [0, 1, 1, 0, 1],
           [0, 1, 1, 1, 0],
           [1, 0, 0, 0, 0],
           [1, 0, 0, 1, 1],
           [1, 0, 1, 0, 1],
           [1, 0, 1, 1, 0],
           [1, 1, 0, 0, 1],
           [1, 1, 0, 1, 0],
           [1, 1, 1, 0, 0],
           [1, 1, 1, 1, 1]])

output_train = array([[1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [1],
           [0],
           [0],
           [0],
           [0],
           [0],
           [0],
           [0],
           [0],
           [0],
           [0],
           [0],
           [0],
           [0],
           [0],
           [0],
           [0]])

random.seed(1)

# randomly initialize our weights with mean 0
weights0 = 2 * random.random((5, 32)) - 1
weights1 = 2 * random.random((32, 1)) - 1



#print "Training network, please wait..."

for j in xrange(10000):
    # Animation 
    animation = "|/-\\"
    sys.stdout.write("\rTraining network, please wait..." + animation[j % len(animation)])
    sys.stdout.flush()
    # End animation

    # Feed forward through layers 0, 1, and 2
    layer0 = input_train
    layer1 = sigmoid(dot(layer0, weights0))
    layer2 = sigmoid(dot(layer1, weights1))

    # how much did we miss the target value?
    layer2_error = output_train - layer2

    # if (j% 10000) == 0:
    #    print "Error:" + str(np.mean(np.abs(layer2_error)))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    layer2_delta = layer2_error * sigmoid(layer2, der=True)

    # how much did each layer1 value contribute to the layer2 error (according to the weights)?
    layer1_error = layer2_delta.dot(weights1.T)

    # in what direction is the target layer1?
    # were we really sure? if so, don't change too much.
    layer1_delta = layer1_error * sigmoid(layer1, der=True)

    weights1 += layer1.T.dot(layer2_delta)
    weights0 += layer0.T.dot(layer1_delta)

print "\nDone training\n"
####----End of training----#####

print "Input: \t\t Output:"

# Passing input hoping network will gues output
inpt = array([0, 0, 0, 0, 0]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([0, 0, 0, 1, 1]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([0, 0, 1, 0, 1]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([0, 0, 1, 1, 0]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([0, 1, 0, 0, 1]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([0, 1, 0, 1, 0]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([0, 1, 1, 0, 0]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([0, 1, 1, 1, 1]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([1, 0, 0, 0, 1]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([1, 0, 0, 1, 0]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([1, 0, 1, 0, 0]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([1, 0, 1, 1, 1]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([1, 1, 0, 0, 0]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([1, 1, 0, 1, 1]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([1, 1, 1, 0, 1]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([1, 1, 1, 1, 0]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([0, 0, 0, 0, 1]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([0, 0, 0, 1, 0]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([0, 0, 1, 0, 0]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([0, 0, 1, 1, 1]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([0, 1, 0, 0, 0]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([0, 1, 0, 1, 1]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([0, 1, 1, 0, 1]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([0, 1, 1, 1, 0]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([1, 0, 0, 0, 0]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([1, 0, 0, 1, 1]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([1, 0, 1, 0, 1]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([1, 0, 1, 1, 0]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([1, 1, 0, 0, 1]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([1, 1, 0, 1, 0]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([1, 1, 1, 0, 0]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================
inpt = array([1, 1, 1, 1, 1]) 

# Use trained network
intermid = sigmoid(dot(inpt, weights0))
result = sigmoid(dot(intermid, weights1))

print inpt, "\t", result[0]
#=========================================


