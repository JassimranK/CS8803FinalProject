import sys
from math import *
import random
from matrix import *
#import numpy as np
def collision_detection(x0, y0):
    dist_top = abs(0.0102 * x0 + y0 - 953.72) / (sqrt(0.0102 ** 2 + 1 ** 2))
    dist_bot = abs(0.0102 * x0 + y0 - 145) / (sqrt(0.0102 ** 2 + 1 ** 2))
    dist_left = abs(801 * x0 - 7 * y0 - 220835) / (sqrt(801 ** 2 + 7 ** 2))
    dist_right = abs(10 * x0 - 1363 * y0 + 197699) / (sqrt(10 ** 2 + 1363 ** 2))
    retval = False
    if dist_top < 10 or dist_bot < 10 or dist_left < 10 or dist_right < 10:
        retval = True
    return retval
        #TODO: If it happend in last 60: 1) turn 180 2) or calculate angle

def measurement_update(P, x, z):
    if collision_detection(z[0], z[1]):
        x = matrix([[0.], [0.], [0.], [0.]]) # initial state (location and velocity)
        P = matrix([[10., 0., 0., 0.], [0., 10., 0., 0.], [0., 0., 500., 0.], [0., 0., 0., 500.]]) # initial
    
    Hx = H * x
    zMatrix = matrix([[z[0]], [z[1]]])
    y = zMatrix - Hx
    S = H * P * matrix.transpose(H) + R
    K = P * matrix.transpose(H) * matrix.inverse(S)
    x = x + (K * y)
    P = (I - K * H) * P
    return P, x

def kalman_filter(x, P):

    lastPrediction = None

    for n in range(len(measurements)):
        
        #TODO Update the filter to support calculating both X and Y and update
        #this line as needed
        z = measurements[n]

        # measurement update
        P, x = measurement_update(P, x, z)

        # prediction
        x = F * x + u
        P = F * P * matrix.transpose(F)

        lastPrediction = [x.value[0][0], x.value[1][0]]
    
    for i in range(60):
        z = lastPrediction

        # measurement update
        P, x = measurement_update(P, x, z)

        # prediction
        x = F * x + u
        P = F * P * matrix.transpose(F)

        #TODO Update to store the last Y prediction as well.
        lastPrediction = [x.value[0][0], x.value[1][0]]
        #TODO This should be changed to append the real Y value.  Right now it
        #is set to duplicate the X value until the filter supports more
        #dimensions
        predictions.append([x.value[0][0], x.value[1][0]])


    return x,P
	
dt = 1.0/30 	#30 frames per second
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
filename = sys.argv[1]
#filename = "inputs/test00.txt"
linesOfFile = open(filename, 'r').readlines()
measurements = []
predictions = []

for line in linesOfFile:
    xValue, yValue = line.rstrip('\n').split(',')
    measurements.append([float(xValue), float(yValue)])
#    pass


print(kalman_filter(x, P))

with open('prediction.txt', 'w') as f:
    for prediction in predictions:
        print >> f, '%s,%s' % (int(round(prediction[0],0)),int(round(prediction[1],0)))
    #for prediction in predictions:
     #   print('%s,%s' % (prediction[0], prediction[1]), end="\n", file=f)
    #for _ in range(60):
    #    print >> f, '%s,%s' % (prediction[0].strip(), prediction[1].strip())