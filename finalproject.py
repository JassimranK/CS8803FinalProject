import sys
from math import *
import random
from matrix import *

def measurement_update(P, x, z):
    y = matrix([[z]]) - H * x
    S = H * P * matrix.transpose(H) + R
    K = P * matrix.transpose(H) * matrix.inverse(S)
    x = x + (K * y)
    P = (I - K * H) * P
    return P, x

def kalman_filter(x, P):

    lastPrediction = 0

    for n in range(len(measurements)):
        
        #TODO Update the filter to support calculating both X and Y and update this line as needed
        z = measurements[n][0]

        # measurement update
        P, x = measurement_update(P, x, z)

        # prediction
        x = F * x + u
        P = F * P * matrix.transpose(F)

        lastPrediction = x.value[0][0]
    
    for i in range(60):
        z = lastPrediction

        # measurement update
        P, x = measurement_update(P, x, z)

        # prediction
        x = F * x + u
        P = F * P * matrix.transpose(F)

        #TODO Update to store the last Y prediction as well.
        lastPrediction = x.value[0][0]
        #TODO This should be changed to append the real Y value.   Right now it is set to duplicate the X value until the filter supports more dimensions
        predictions.append([x.value[0][0], x.value[0][0]])


    return x,P

x = matrix([[0.], [0.]]) # initial state (location and velocity)
P = matrix([[1000., 0.], [0., 1000.]]) # initial uncertainty
u = matrix([[0.], [0.]]) # external motion
F = matrix([[1., 1.], [0, 1.]]) # next state function
H = matrix([[1., 0.]]) # measurement function
R = matrix([[1.]]) # measurement uncertainty
I = matrix([[1., 0.], [0., 1.]]) # identity matrix



#filename = sys.argv[1]
filename = "inputs/test00.txt"
linesOfFile = open(filename, 'r').readlines()
measurements = []
predictions = []

for line in linesOfFile:
    xValue, yValue = line.rstrip('\n').split(',')
    measurements.append([float(xValue), float(yValue)])
    pass

print(kalman_filter(x, P))

with open('prediction.txt', 'w') as f:
    for prediction in predictions:
        print('%s,%s' % (prediction[0], prediction[1]), end="\n", file=f)
    #for prediction in predictions:
    #    print >> f, '%s,%s' % (prediction[0], prediction[1])
    #for _ in range(60):
    #    print >> f, '%s,%s' % (x.strip(), y.strip())
