'''
Author: Alex Rutherford
'''

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib import patches 
import time
import datetime

from numpy.core.defchararray import array
#import herding_algorithm

#yet to add the angle for neighbourhood, just distance based
#flocking 'strength' is constant over visual range, perhaps should decrease as distance increases
#need to add a minimum speed check (birds cannot fly with zero velocity)
# should probably make another class to containing plotting functions and then inheret methods into boid and predator classes

def distance(x, y):
    ''' Euclidean distance between two points '''
    return np.linalg.norm(np.array(x) - np.array(y))

""" This is new ----------------------------------"""
def calculateDivPos(flock, turbines):
  #Calculate the divergence position based on current flock speed direction and turbine positions
    xMin = 100000
    xMax = -1
    yMax = -1
    for pos in turbines:
        if(pos[0]>xMax):
            xMax = pos[0]
        if(pos[0]<xMin):
            xMin = pos[0]
        if(pos[1]>yMax):
            yMax = pos[1]

    flockCG = FlockCG(0,0)
    flockCG.formCG(flock)
    if(abs(xMin-flockCG.pos[0])<abs(xMax-flockCG.pos[0])):
        posDiv = np.array([xMin-200, yMax+50])
    else:
        posDiv = np.array([xMax+200, yMax+50])
    return posDiv


class FlockCG():
    def __init__(self, pos, velocity):
        self.pos = pos
        self.velocity = velocity

    def formCG(self, flock):
        pos = np.array([0.0,0.0])
        vel = np.array([0.0,0.0])
        for boid in flock:
            pos+= boid.pos
            vel+= boid.velocity
        self.pos = pos/len(flock)
        self.velocity = vel/len(flock)

"""-------------------------------------------------"""

class Boid():
    ''' class to hold information for each boid/bird
    Args:
        pos = initial position of boid
        velocity = initial velocity of boid
        visual = sets the neighorhood distance
        flee = range at which boid moves away from predator while maintaing flocking behaviour
        urgent = range at which boid enters selfish escape mode
        maxSpeed = maximum speed of the boid
        minDist = minimum distance other boids are meant to be
    '''

    def __init__(self,
                pos,
                velocity,
                visual = 300,
                flee = 200,
                urgent = 50,
                maxSpeed = 5,
                minDist = 30):
        
        self.pos = pos
        self.velocity = velocity

        #constants
        self.visual = visual
        self.flee = flee
        self.urgent = urgent
        self.maxSpeed = maxSpeed
        self.minDist = minDist

        #for plotting
        self.patch = None #patches.Polygon (triangle) object associated with the boid
        self.tri = None #coordiates of the triangle
        

    def flocking(self, flock):
        ''' Implements the cohesion condition - steer to move towards the average position of flockmates
        Args:
            flock: list of all boid objects
        '''
        #flocking weights -- adjust velocity by this %
        centering = 0.005
        avoid = 0.05
        matching = 0.05

        num_neighbours = 0
        centre = np.zeros(2) #for centering
        avg_vel = np.zeros(2) #for matching
        move = np.zeros(2) #for avodiing flockmates
        for boid in flock:
            if boid == self:
                continue

            #centering & matching velocity
            if distance(self.pos, boid.pos) < self.visual:
                num_neighbours += 1
                centre += boid.pos
                avg_vel += boid.velocity

                #avoid 
                if distance(self.pos, boid.pos) < self.minDist:
                    move += self.pos - boid.pos
        
        if num_neighbours != 0:
            centre = centre / num_neighbours
            avg_vel = avg_vel / num_neighbours
            self.velocity += (centre - self.pos) * centering + (avg_vel - self.velocity) * matching + move * avoid

    def checkPredators(self, predators):
        ''' Implements a basic sense of predator avoidance based on two distance criterions
        Args:
            predators: list of all predator objects
        '''
        predator_weight = 0.75 #adjust velocity by this %

        for predator in predators:

            dist = distance(self.pos, predator.pos)
            '''
            if dist <= self.urgent: #take selfish action
            
                #perpendicular velocity vector --> may be better to use angle..? but maybe better to use vectors for 3d transition..?
                vperp = np.array([-predator.velocity[1], predator.velocity[0]])
                vperp = vperp/np.linalg.norm(vperp) #normalize

                #vector from pred to bird
                pb = np.array([self.pos[0]-predator.pos[0], self.pos[1]-predator.pos[1]])
                sign = 1
                if np.dot(vperp, pb/np.linalg.norm(pb)) < 0: #determine direction of escape
                    sign = -1
                    vperp = -1*vperp

                new_angle = self.get_angle() + np.pi/4 * sign * (1 - np.dot(vperp, self.velocity/np.linalg.norm(self.velocity)))
                speed = np.linalg.norm(self.velocity)
                self.velocity = np.array([np.cos(new_angle)*speed, np.sin(new_angle)*speed])
                return new_angle #exit function as urgent avoidance will take priority --> issue if two predators are within urgent range, maybe rank predators by distance before this operation
            elif dist <= self.flee:
            '''
            if dist <= self.flee:
                #evasive action within flock
                self.velocity += (self.pos - predator.pos) * predator_weight
            else:
                continue #do nothing        
            return

    def updatePosition(self):
        ''' Updates the boids position using velocity
        Implements a maximum speed check (birds can only fly so fast)
        '''
        speed = np.linalg.norm(self.velocity)
        if speed > self.maxSpeed:
            self.velocity = (self.velocity/speed) * self.maxSpeed

        self.pos += self.velocity

    def get_angle(self):
        ''' Returns the angle of the boids's velocity '''
        return np.arctan2(self.velocity[1], self.velocity[0])

    def tri_coords(self):
        ''' For plotting -- returns the coordinates for the triangle using get_angle()
        '''
        angle_adjustment = 0.26 # size of angle at top of triangle
        side_length = 5 # how long triangle sides are
        angle = self.get_angle()
        x1 = [self.pos[0]-side_length*np.cos(angle-angle_adjustment),self.pos[1]-5*np.sin(angle-angle_adjustment)]
        x2 = [self.pos[0]-side_length*np.cos(angle+angle_adjustment),self.pos[1]-5*np.sin(angle+angle_adjustment)]
        self.tri = [self.pos, x1, x2]


class Predator():
    ''' class to hold information for each predator/
    Args:
        pos = initial position of predator
        velocity = initial velocity of predator
        maxSpeed = maximum speed the predator can travel at
    '''
    def __init__(self,
                pos,
                velocity=np.zeros(2),
                maxSpeed = 10):
        self.pos = pos
        self.velocity = velocity
        self.maxSpeed = maxSpeed

        #for plotting
        self.patch = None #patches.Polygon (triangle) object associated with the boid
        self.radius = None #patches.Circle object to show the flee range
        self.tri = None #coordiates of the triangle

    
    def huntClosest(self, flock):
        ''' Predator steers towards the closest boid to 'hunt'
        '''
        hunt_weight = 1 #adjust velocity by this %

        closest = flock[0]
        min_dist = distance(self.pos, closest.pos)
        for boid in flock:
            if distance(self.pos, boid.pos) < min_dist:
                closest = boid
                min_dist = distance(self.pos, boid.pos)

        self.velocity += (closest.pos - self.pos) * hunt_weight

    """ This is new -------------------------------------- """

    def calculateRo(self, flock, x, posDiv, sign):
        #calculate deviation
        """ Sign +- 1 indicates which point for deviation we've chosen"""
        predator_weight = 0.75 #adjust velocity by this %
        """ Fp = x*predator_weight"""
        r_dc = posDiv - flock.pos
        vel_div = flock.velocity + (flock.pos - x) * predator_weight
        ro = sign*np.dot(vel_div,r_dc)
        """
        print(r_dc, kg*predator.velocity, Fp)
        ro = np.cross(r_dc, kg*predator.velocity) + np.cross(r_dc, Fp) """
        return ro

    #SELECT M WAYPOINT ALGORITHM
    def selectPoints(self, flockCG):
        """Define permissable persuer positions, range Xp_min - Xp_max"""
        maxSampleNumber = 10 
        m = 1 #desired points
        Xpm = np.array([])
        #Gp = np.array([])
        ro_min = 83798473

        min = [self.pos[0]-60, self.pos[0]-60]
        max = [self.pos[1]+60, self.pos[1]]
        x = np.random.uniform(low = min, high=max, size=(maxSampleNumber, 2))
        for xs in x:
            ro = self.calculateRo(flockCG, xs, posDiv, +1)
            if(ro<ro_min):
                Xpm = xs
                ro_min = ro
            
        """zipped_pairs = zip(Gp, Xpm)
        z = [var for _, var in sorted(zipped_pairs, reverse=False)]
        if len(z) > m :
            Xpm = Xpm[0:m-1]"""

        return Xpm

    def flyMode(self, node_pos):
        #return velocity vector
        intensity = np.linalg.norm(self.velocity)
        scale = intensity/distance(node_pos, self.pos)
        up = np.array([scale*(node_pos[0]-self.pos[0]), scale*(node_pos[1]-self.pos[1])])
        if(np.linalg.norm(up)>self.maxSpeed):
            up = up*self.maxSpeed/np.linalg.norm(up)
        self.velocity = up
        """
        flyTime = distance(node_pos, flock) / predator.velocity
        fly(up, flyTime, predator) """

    def engageMode(self, node_pos, flock):
        #return velocity vector
        vect = np.array([flock.pos[0] - node_pos[0], flock.pos[1] - node_pos[1]])
        """ We fly perpendicular to this """
        up = 0.5*np.transpose(vect)
        if(np.linalg.norm(up)>self.maxSpeed):
            up = up*self.maxSpeed/np.linalg.norm(up)
        self.velocity = up  

    def herdingAlg(self, flock):
        #tau = 3 #predetermined duration of engage mode, seconds
        dist_treshdold = 0 #epsilon from their code, if further than this fly directly
        #t_last = time.time()

        flockCG = FlockCG(0,0)
        flockCG.formCG(flock)
        
        updateWaypointSet = True
        #if(flockCG.velocity[1]/flockCG.velocity[0] < (posDiv[1]-flockCG.pos[1])/(posDiv[0]-flockCG.pos[0])) or (posDiv[1]>flockCG.pos[1]):
        if(True):
            Xpm = []
            if updateWaypointSet == True :
                Xpm = self.selectPoints(flockCG) #Call select m waypoints algorithm
                t_last = time.time()
                updateWaypointSet = False

            while (updateWaypointSet == False):
                if(distance(self.pos, Xpm)>dist_treshdold):
                    self.flyMode(Xpm)
                else:
                    self.engageMode(Xpm, flockCG)
                updateWaypointSet = True
                

        """ ------------------------------------------------- """

    def updatePosition(self):
        ''' Updates the predators position using velocity
        Implements a maximum speed check (birds can only fly so fast)
        '''
        speed = np.linalg.norm(self.velocity)
        if speed > self.maxSpeed:
            self.velocity = (self.velocity/speed) * self.maxSpeed
        self.pos += self.velocity

    def get_angle(self):
        ''' Returns the angle of the predator's velocity '''
        return np.arctan2(self.velocity[1], self.velocity[0])

    def tri_coords(self):
        ''' For plotting -- returns the coordinates for the triangle using get_angle()
        '''
        angle_adjustment = 0.26 # size of angle at top of triangle
        side_length = 10 # how long triangle sides are
        angle = self.get_angle()
        x1 = [self.pos[0]-side_length*np.cos(angle-angle_adjustment),self.pos[1]-5*np.sin(angle-angle_adjustment)]
        x2 = [self.pos[0]-side_length*np.cos(angle+angle_adjustment),self.pos[1]-5*np.sin(angle+angle_adjustment)]
        self.tri = [self.pos, x1, x2]

if __name__ == '__main__':

    #Constants
    num_boids = 40 #number of boids
    num_preds = 1 #number of predators
    visual_range = 300 #neighbourhood range
    flee_range = 200 #range at which boids will evade a predator
    urgent_range = 50 #range at which selfish evasive action is taken
    max_speed = 5
    max_acc = 5

    limits = [1000, 1000] #axes limits

    #Turbines
    turbines = np.array([[300, 5], [300, 205], [500, 5], [500, 205], [700, 100]])

    #generate boids and predators (with random positions and velocities)
    flock = [Boid((np.random.rand(2)*400+500), np.random.rand(2)*5) for _ in range(num_boids)] #generate boid objects
    #flock = [Boid(np.array([675.5, 675.5]), np.random.rand(2)*5)]

    posDiv = calculateDivPos(flock, turbines) #divergence position where we want to take birds

    predator_pos = np.array([1000.5,700.5])
    predators = [Predator(predator_pos,  np.random.rand(2)*5) for _ in range(num_preds)]

    #create figure and set axes limits
    figure = plt.figure()
    axes = plt.axes(xlim=[0,limits[0]], ylim = [0,limits[1]])

    #plot initial turbine positions
    blade_length = 36
    for turbine in turbines:
        axes.scatter(turbine[0], turbine[1], c='g', marker='+')
        axes.add_patch(patches.Circle(turbine, radius=blade_length, ec='g', fill=False))

    #plot initial position of flock
    for boid in flock:
        boid.tri_coords() #get angle and determine coordinates of triangles
        boid.patch = patches.Polygon(boid.tri, closed=True, fc='b', ec='b') #create patch object
        axes.add_patch(boid.patch) #add patch to graph

    #plot initial positions of predators
    for predator in predators:
        predator.tri_coords() #determine coords of triangle
        predator.patch = patches.Polygon(predator.tri, closed=True, fc='r', ec='r') #create patch object
        axes.add_patch(predator.patch) #add patch object to graph

        #plot flee radius --> probably temporary
        predator.radius = patches.Circle(predator.pos, radius=flee_range, ec='red', fill=False)
        axes.add_patch(predator.radius)

    #function called at each animation frame
    def animate(frame):
        for boid in flock:
            '''
            boid.flyTowardsCenter(visual_range, flock)
            boid.avoidOthers(flock)
            boid.matchVelocity(visual_range, flock)
            '''
            boid.flocking(flock)
            alpha = boid.checkPredators(predators)
            boid.updatePosition()

            #update patches
            angle = np.arctan2(boid.velocity[1], boid.velocity[0])
            x1 = [boid.pos[0]-10*np.cos(angle-0.26),boid.pos[1]-10*np.sin(angle-0.26)]
            x2 = [boid.pos[0]-10*np.cos(angle+0.26),boid.pos[1]-10*np.sin(angle+0.26)]
            boid.tri = [boid.pos, x1, x2]
            boid.patch.set_xy(boid.tri)

            #ignore this -- > for selfish escape
            '''
            if alpha is not None:
                dx = 50 * np.cos(alpha)
                dy = 50 * np.sin(alpha)
                plt.arrow(boid.pos[0], boid.pos[1], dx, dy, alpha=0.3)
            '''
        for predator in predators:
            predator.herdingAlg(flock)
            predator.updatePosition()

            #update patches
            angle = np.arctan2(predator.velocity[1], predator.velocity[0])
            x1 = [predator.pos[0]-10*np.cos(angle-0.26),predator.pos[1]-10*np.sin(angle-0.26)]
            x2 = [predator.pos[0]-10*np.cos(angle+0.26),predator.pos[1]-10*np.sin(angle+0.26)]
            predator.tri = [predator.pos, x1, x2]
            predator.patch.set_xy(predator.tri)
            predator.radius.center = predator.pos

    #run animation
    anim = animation.FuncAnimation(figure, animate,
                                    frames = 50, interval = 50)

    plt.show()
    anim.save('sin.mp4')

 