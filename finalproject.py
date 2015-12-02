import sys
from math import *
from matrix import *
import pylab as pl

# collision_detection is a function used to determine which side of the box the robot has hit
def collision_detection(x0, y0):
    # Use the equation of the line defining the top and bottom of the box to 
    # calculate the distance of the robot from the boundary of the box
    dist_top = abs(0.0102 * x0 + y0 - 953.72) / (sqrt(0.0102 ** 2 + 1 ** 2))
    dist_bot = abs(0.0102 * x0 + y0 - 145) / (sqrt(0.0102 ** 2 + 1 ** 2))
    
    retval = "NONE"
    # If the coordinate is within 10 units of the boundary we deterime that a collision has occurred
    # As a safeguard we check to see if the value exceeds a maximum to cover cases where the distance 
    # jumps the 10 unit buffer zone
    if dist_top < 10 or y0 > 974:
        retval = "TOP"
    if dist_bot < 10 or y0 < 105:
        retval = "BOTTOM"
    # The left and right walls of the box are close enough to vertical to be defined by a hard-coded
    # min an max value for the X coordinate
    if x0 < 240:
        retval = "LEFT"
    if x0 > 1696:
        retval = "RIGHT"
    return retval
        
# ComputeAvgDistance is used to calculate the average distance between the set of measurments provided
def ComputeAvgDistance(measurements):
    totalDistance = 0.
    numberOfMeasurements = len(measurements[-lookBackFrames:])
    for i in range(numberOfMeasurements):
        # Accrue the distance between the points
        totalDistance += distanceBetween(measurements[i], measurements[i - 1])
    # return the average distance
    return totalDistance / numberOfMeasurements

# distanceBetween calculates the distance between two given points
def distanceBetween(point1, point2):
   x1, y1 = point1
   x2, y2 = point2
   return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# measurement_update performs the required calculations for the KF to incorporate a new measurment into its model
def measurement_update(P, x, z):
    Z = matrix([[z[0]], [z[1]]])
    Hx = (H * x)
    y = Z - Hx
    S = H * P * H.transpose() + R
    K = P * H.transpose() * S.inverse()
    x = x + (K * y)
    P = (I - (K * H)) * P
    return P, x

# predict determines the estimated next location of the robot
def predict(P, x):
    # the reInitialized flag determines if we should start with fresh x and P matrices
    reInitialized = False

    #Since the KF only works with linear problems, we can turn it into a simple
    #linear problem by only looking back at recent measurments / predictions
    for measurement in measurements[-lookBackFrames:]:
        if reInitialized == False:
            reInitialized = True
            # use the current measurement as a starting point for the x matrix
            x = matrix([[measurement[0]], [measurement[1]], [0.], [0.]])
            # the location is well known, but the velocity is not, so initialize the P matrix accordingly
            P = matrix([[10., 0., 0., 0.], [0., 10., 0., 0.], [0., 0., 500., 0.], [0., 0., 0., 500.]])

        z = measurement

        # determine if the current measurement results in a collision
        collResult = collision_detection(z[0], z[1])
        if collResult != "NONE":
            # if there has been a collision, use the current measurement and velocities to initialize the x matrix
            x = matrix([[z[0]], [z[1]], [x.value[2][0]], [x.value[3][0]]])
            # In this case, the location and valocity are both relatively well known, so initialize P accordingly
            P = matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 10., 0.], [0., 0., 0., 10.]])
            
            avgDist = ComputeAvgDistance(measurements)

            # use the average distance between recent measurements to manually calculate the next predicted location to be used as an input into the KF
            if collResult == "TOP":
                z = [measurement[0], measurement[1] - avgDist]
            if collResult == "BOTTOM":
                z = [measurement[0], measurement[1] + avgDist]
            if collResult == "LEFT":
                z = [measurement[0] + avgDist, measurement[1]]
            if collResult == "RIGHT":
                z = [measurement[0] - avgDist, measurement[1]]

        # incorporate the measurment into the KF
        P, x = measurement_update(P, x, z)
    
        # draw a prediction from the KF
        x = F * x + u
        P = F * P * matrix.transpose(F)
    
    # set and return the lastPrediction for further processing
    lastPrediction = [x.value[0][0], x.value[1][0]]
    return lastPrediction

# our implementation of the KF that has been set to produce 60 predictions of the robot's location
def kalman_filter(x, P):

    for i in range(60):

        lastPrediction = predict(P, x)
        # append the last prediction to the list of measurments so that it can be used 
        # as the basis for further predictions
        measurements.append(lastPrediction)
        # append the last prediction to the list of predictions so that they can later be written to a file
        predictions.append(lastPrediction)

    return predictions

global x 
# initial state (location and velocity)
x = matrix([[0.], [0.], [0.], [0.]]) 
global P 
# initial uncertinaty
P = matrix([[10., 0., 0., 0.], [0., 10., 0., 0.], [0., 0., 500., 0.], [0., 0., 0., 500.]]) 
# external motion
u = matrix([[0.], [0.], [0.], [0.]]) 
# state transiton matrix
F = matrix([[1., 0., 1., 0.], [0., 1., 0., 1.], [0., 0., 1., 0.], [0, 0., 0., 1.]]) 
# measurement function
H = matrix([[1., 0., 0., 0.], [0., 1., 0., 0.]]) 
# measurement uncertainty
R = matrix([[1.0, 0.], [0., 1.0]])
# identity matrix
I = matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

filename = sys.argv[1]

# open and read the lines of the input file
linesOfFile = open(filename, 'r').readlines()
# defines the number of measurements to consider for our Fading Memory Filter Implementation
lookBackFrames = 4
# contains the input measurements (our predictions will be appended here as well)
measurements = []
# contains only our predictions of the robot's location
predictions = []


for line in linesOfFile:
    xValue, yValue = line.rstrip('\n').split(',')
    measurements.append([float(xValue), float(yValue)])
    
# call the KF
print(kalman_filter(x, P))

# write the predictions to a file
with open('prediction.txt', 'w') as f:
    print 'length of predictions', len(predictions)
    for prediction in predictions:
        print >> f, '%s,%s' % (prediction[0],prediction[1])