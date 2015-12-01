import sys
from math import *
import random
from matrix import *
import numpy as np
import pylab as pl

def collision_detection(x0, y0):
    dist_top = abs(0.0102 * x0 + y0 - 953.72) / (sqrt(0.0102 ** 2 + 1 ** 2))
    dist_bot = abs(0.0102 * x0 + y0 - 145) / (sqrt(0.0102 ** 2 + 1 ** 2))
    dist_left = abs(801 * x0 - 7 * y0 - 220835) / (sqrt(801 ** 2 + 7 ** 2))
    dist_right = abs(10 * x0 - 1363 * y0 + 197699) / (sqrt(10 ** 2 + 1363 ** 2))
    retval = "NONE"
    if dist_top < 10 or y0 > 974:
        retval = "TOP"
    if dist_bot < 10 or y0 < 105:
        retval = "BOTTOM"
    if  dist_left < 10 or x0 < 240:
        retval = "LEFT"
    if dist_right < 10 or x0 > 1696:
        retval = "RIGHT"
    return retval
        #TODO: If it happend in last 60: 1) turn 180 2) or calculate angle

def ComputeAvgDistance(measurements):
    totalDistance = 0.
    numberOfMeasurements = len(measurements[-lookBackFrames:])
    for i in range(numberOfMeasurements):
        totalDistance += distanceBetween(measurements[i], measurements[i - 1])
    return totalDistance / numberOfMeasurements


def distanceBetween(point1, point2):
   """Computes distance between point1 and point2. Points are (x, y) pairs."""
   x1, y1 = point1
   x2, y2 = point2
   return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def measurement_update(P, x, z):
        ##################
        # prediction
    #x = (F * x) + u
    #P = F * P * F.transpose()
        
    # measurement update
    Z = matrix([[z[0]], [z[1]]])
    Hx = (H * x)
    y = Z - Hx
    S = H * P * H.transpose() + R
    K = P * H.transpose() * S.inverse()
    x = x + (K * y)
    P = (I - (K * H)) * P
        #################################
    #Hx = H * x
    #zMatrix = matrix([[z[0]], [z[1]]])
    #y = zMatrix - Hx
    #S = H * P * matrix.transpose(H) + R
    #K = P * matrix.transpose(H) * matrix.inverse(S)
    #x = x + (K * y)
    #P = (I - K * H) * P
    return P, x

def predict(P, x):
    #Since the KF only works with linear problems, we can turn it into a simple
    #linear problem by only looking back at recent measurments / predictions

    reInitialized = False

    for measurement in measurements[-lookBackFrames:]:
        if reInitialized == False:
            reInitialized = True
            x = matrix([[measurement[0]], [measurement[1]], [0.], [0.]])
            P = matrix([[10., 0., 0., 0.], [0., 10., 0., 0.], [0., 0., 500., 0.], [0., 0., 0., 500.]])
        

        ##################################################
        z = measurement

        collResult = collision_detection(z[0], z[1])
        if collResult != "NONE":
            x = matrix([[measurement[0]], [measurement[1]], [x.value[2][0]], [x.value[3][0]]])
            P = matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 10., 0.], [0., 0., 0., 10.]])
            avgDist = ComputeAvgDistance(measurements)
            if collResult == "TOP":
                z = [measurement[0], measurement[1] - avgDist]
            if collResult == "BOTTOM":
                z = [measurement[0], measurement[1] + avgDist]
            if collResult == "LEFT":
                z = [measurement[0] + avgDist, measurement[1]]
            if collResult == "RIGHT":
                z = [measurement[0] - avgDist, measurement[1]]

        ##################################################


        # measurement update
        P, x = measurement_update(P, x, z)
    
        # prediction
        x = F * x + u
        P = F * P * matrix.transpose(F)
    
    lastPrediction = [x.value[0][0], x.value[1][0]]
    return lastPrediction

def kalman_filter(x, P):

    for i in range(60):

        lastPrediction = predict(P, x)
        measurements.append(lastPrediction)
        predictions.append(lastPrediction)

    return predictions


dt = 1.0#/30 #30 frames per second
global x 
x = matrix([[0.], [0.], [0.], [0.]]) # initial state (location and velocity)
global P 
#x = matrix([[0.], [0.], [0.], [0.]]) # initial state (location and velocity)
P = matrix([[10., 0., 0., 0.], [0., 10., 0., 0.], [0., 0., 500., 0.], [0., 0., 0., 500.]]) #
u = matrix([[0.], [0.], [0.], [0.]]) # external motion
F = matrix([[1., 0., dt, 0.], [0., 1., 0., dt], [0., 0., 1., 0.], [0, 0., 0., 1.]]) # next state function
H = matrix([[1., 0., 0., 0.], [0., 1., 0., 0.]]) # measurement function
                                                 # measurement uncertainty
 #TODO: try 1.0
R = matrix([[1.0, 0.], [0., 1.0]])
# identity matrix
I = matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

#filename = sys.argv[1]
filename = "inputs/test05.txt"
linesOfFile = open(filename, 'r').readlines()
lookBackFrames = 4
measurements = []
predictions = []
mB = []
nB = []
for line in linesOfFile:
    xValue, yValue = line.rstrip('\n').split(',')
    measurements.append([float(xValue), float(yValue)])
    mB.append([float(xValue)])
    nB.append(float(yValue))
#    pass

print(kalman_filter(x, P))
pl.title('Plot prediction')
pl.plot(mB,nB, 'b')
#pl.xlim(0.0, 1800)
#pl.ylim(0.0, 1000)
m = []
n = []
mA = []
nA = []

with open('prediction.txt', 'w') as f:
    print 'length of predictions', len(predictions)
    for prediction in predictions:
        #pl.plot(int(round(prediction[0],0)), int(round(prediction[1],0)))
        #print >> f, '%s,%s' %
        #(int(round(prediction[0],0)),int(round(prediction[1],0)))
        print >> f, '%s,%s' % (prediction[0],prediction[1])
        m.append(int(round(prediction[0],0)))
        n.append(int(round(prediction[1],0)))

pl.plot(m,n, 'r')

filename = "actual/05.txt"
linesOfFileA = open(filename, 'r').readlines()

for line in linesOfFileA:
    xValue, yValue = line.rstrip('\n').split(',')
    mA.append([float(xValue)])
    nA.append(float(yValue))

pl.plot(mA,nA, 'g')
pl.show() 
	#for prediction in predictions:
     #   print('%s,%s' % (prediction[0], prediction[1]), end="\n", file=f)
    #for _ in range(60):
    #    print >> f, '%s,%s' % (prediction[0].strip(), prediction[1].strip())