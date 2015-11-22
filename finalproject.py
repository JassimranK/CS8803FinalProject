import sys
from math import *
import random
from matrix import *
import numpy as np


def measurement_update(P, x, z):
    Hx = H * x
    zMatrix  = matrix([[z[0]], [z[1]]])
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
        Q = matrix([[1/4.0*dt**4 , 1/4.0*dt**4, 1/2.0*dt**3, 1/2.0*dt**3, 1/2.0*dt**2, 1/2.0*dt**2], 
		       [1/4.0*dt**4 , 1/4.0*dt**4, 1/2.0*dt**3, 1/2.0*dt**3, 1/2.0*dt**2, 1/2.0*dt**2],
               [1/2.0*dt**3 , 1/2.0*dt**3, dt**2, dt**2, dt, dt],
			   [1/2.0*dt**3 , 1/2.0*dt**3, dt**2, dt**2, dt, dt],
			   [1/2.0*dt**2 , 1/2.0*dt**2, dt, dt, 1.0, 1.0],
               [1/2.0*dt**2 , 1/2.0*dt**2, dt, dt, 1.0, 1.0]])
        
        P = F * P * matrix.transpose(F) 

        #TODO Update to store the last Y prediction as well.
        lastPrediction = [x.value[0][0], x.value[1][0]]
        #TODO This should be changed to append the real Y value.  Right now it
        #is set to duplicate the X value until the filter supports more
        #dimensions
        predictions.append([x.value[0][0], x.value[1][0]])


    return x,P
	
dt = 1.0/30 	#30 frames per second

x = matrix([[0.], [0.], [0.], [0.], [0.], [0.]]) # initial state (location and velocity)
P = matrix([[1000., 0., 0., 0., 0. ,0.], [0., 1000., 0., 0., 0. ,0.], [0., 0., 1000., 0., 0. ,0.], [0., 0., 0., 1000., 0. ,0.], [0., 0., 0., 0., 1000., 0.], [0., 0., 0., 0., 0., 1000.]]) # initial uncertainty 1000 for both position and velocity and acc
u = matrix([[0.], [0.], [0.], [0.], [0.], [0.]]) # external motion
F = matrix([[1., 0., dt, 0., 1/2 * dt**2, 0.], [0., 1., 0., dt, 0., 1/2 * dt**2], [0., 0., 1., 0., dt, 0.], [0, 0., 0., 1., 0., dt], [0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 1.]]) # next state function
H = matrix([[1., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.]]) # measurement function
 # measurement uncertainty
 #TODO: try 1.0
R = matrix([[1.0, 0.], [0., 1.0]])
# identity matrix
I = matrix([[1., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0.], [0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 1.]]) 

filename = sys.argv[1]
#filename = "inputs/test001.txt"
linesOfFile = open(filename, 'r').readlines()
measurements = []
predictions = []

for line in linesOfFile:
    xValue, yValue = line.rstrip('\n').split(',')
    measurements.append([float(xValue), float(yValue)])
#    pass



x_p, P_p = (kalman_filter(x, P))
x_p.show()


#f = KalmanFilter(dim_x=4, dim_z=2)
#dt = 1. / 30 #30 frames per second

#f.u = 0.

##Assign the initial value for the state (position and velocity).  You can do
##this with a two dimensional array like so:

##f.x = np.array([[2.], # position
##                [0.]]) # velocity
##or just use a one dimensional array, which I prefer doing.
#f.x = np.array([[0, 0, 0 , 0]]).T #check
##Define the state transition matrix:
#f.F = np.array([[1, dt, 0,  0],
#                [0,  1, 0,  0],
#                [0,  0, 1, dt],
#                [0,  0, 0,  1]]) #check
##Define the measurement function:
#f.H = np.array([[1., 0, 0, 0],
#                [0, 0, 1., 0]]) #check
##Define the covariance matrix.  Here I take advantage of the fact that P
##already contains np.eye(dim_x), and just multipy by the uncertainty:

##f.P *= 1000.
##I could have written:
#f.P = np.eye(4) * 1000. #check
###You decide which is more readable and understandable.

##Now assign the measurement noise.  Here the dimension is 1x1, so I can use a
##scalar

##f.R = 5
##I could have done this instead:
#f.R = np.eye(2) * 5 #check

##Note that this must be a 2 dimensional array, as must all the matrices.

##Finally, I will assign the process noise.  Here I will take advantage of
##another FilterPy library function:
#q = Q_discrete_white_noise(dim=2, dt=dt, var=0.05)
#f.Q = block_diag(q,q) #check
####Now just perform the standard predict/update loop:
###while some_condition_is_true:

##filename = sys.argv[1]
#filename = "inputs/test00.txt"
#linesOfFile = open(filename, 'r').readlines()
#measurements = []
#predictions = []

#for line in linesOfFile:
#    xValue, yValue = line.rstrip('\n').split(',')
#    #measurements = np.array([np.array([xValue, yValue]).T for _ in range()])
#    #measurements.append([float(xValue), float(yValue)])
#    measurements.append(np.array([xValue, yValue]))

#lastPrediction = 0

##f.batch_filter(measurements)

#for n in range(len(measurements)):
        
#    #TODO Update the filter to support calculating both X and Y and update this
#    #line as needed
#    z = measurements[n]

#    f.update(z)

#    f.predict()


#    #### measurement update
#    ###P, x = measurement_update(P, x, z)

#    #### prediction
#    ###x = F * x + u
#    ###P = F * P * matrix.transpose(F)

#    lastPrediction = f.x[0]
    
#for i in range(60):
#    z = lastPrediction

#    f.update(np.array(z))

#    f.predict()

#    lastPrediction = f.x[0]

#    predictions.append([f.x[0][0], f.x[0][1]])

    #### measurement update
    ###P, x = measurement_update(P, x, z)

    #### prediction
    ###x = F * x + u
    ###P = F * P * matrix.transpose(F)

    ####TODO Update to store the last Y prediction as well.
    ###lastPrediction = x.value[0][0]
    ####TODO This should be changed to append the real Y value.  Right now it
    ####is set to duplicate the X value until the filter supports more
    ####dimensions
    ###predictions.append([x.value[0][0], x.value[0][0]])

##z = get_sensor_reading()
##f.predict()
##f.update(z)

##do_something_with_estimate (f.x)

with open('prediction.txt', 'w') as f:
    for prediction in predictions:
        print >> f, '%s,%s' % (prediction[0], prediction[1])
    #for prediction in predictions:
        #print('%s,%s' % (prediction[0], prediction[1]), end="\n", file=f)
    #for _ in range(60):
    #    print >> f, '%s,%s' % (x.strip(), y.strip())
