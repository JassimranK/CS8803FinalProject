import sys
from math import *
import random
from matrix import *


def measurement_update(P, x, z):
    ##Hx = H * x
    zMatrix  = matrix([[z[0]], [z[1]]])
    ##y = zMatrix - Hx
    y = zMatrix - x
    S = H * P * matrix.transpose(H) + R
    K = P * matrix.transpose(H) * matrix.inverse(S)
    x = x + (K * y)
    P = (I - K * H) * P
    return P, x

def kalman_filter(x, P):

    lastPrediction = None

    for n in range(len(measurements)):
        
        z = measurements[n]

        # measurement update
        P, x = measurement_update(P, x, z)

        # prediction
        #x = F * x + u
        P = F * P * matrix.transpose(F) + Q

        lastPrediction = [x.value[0][0], x.value[1][0]]
    
    for i in range(60):
        z = lastPrediction

        # measurement update
        P, x = measurement_update(P, x, z)

        # prediction
        x = F * x + u

        #x = matrix([[z[0]+1], [z[1]+2*z[0]]])
        P = F * P * matrix.transpose(F) + Q

        lastPrediction = [x.value[0][0], x.value[1][0]]
        predictions.append([x.value[0][0], x.value[1][0]])


    return x,P

x = matrix([[0.], [0.]]) # initial state (location)
P = matrix([[1000., 0.], [0., 1000.]]) # initial uncertainty
u = matrix([[0.], [0.]]) # external motion
F = matrix([[1.,0.], [2.,1.]]) #
H = matrix([[1., 0.], [0., 1.]]) # measurement function
Q = matrix([[1., 0.], [0., 1.]]) # motion covariance
R = matrix([[.2, 0.], [0., .2]]) # measurement uncertainty
I = matrix([[1., 0.], [0., 1.]]) # identity matrix

##filename = sys.argv[1]
filename = "inputs/test00.txt"
linesOfFile = open(filename, 'r').readlines()
measurements = []
predictions = []

for line in linesOfFile:
    xValue, yValue = line.rstrip('\n').split(',')
    measurements.append([float(xValue), float(yValue)])


print(kalman_filter(x, P))

with open('prediction.txt', 'w') as f:
    for prediction in predictions:
        print >> f, '%s,%s' % (prediction[0], prediction[1])
