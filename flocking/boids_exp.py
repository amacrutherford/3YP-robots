'''
Testing exponential based method
'''

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib import patches 

#yet to add the angle for neighbourhood, just distance based
#flocking 'strength' is constant over visual range, perhaps should decrease as distance increases
#need to add a minimum speed check (birds cannot fly with zero velocity)
# should probably make another class to containing plotting functions and then inheret methods into boid and predator classes

def distance(x, y):
    ''' Euclidean distance between two points '''
    return np.linalg.norm(np.array(x) - np.array(y))

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
                visual = 50,
                flee = 10,
                urgent = 5,
                maxSpeed = 3,
                minDist = 5):
        
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

    def flocking(self, flock, vd, predators):
        ''' Implements the cohesion condition
        Args:
            flock: list of all boid objects
            vd: preselected steady state velocity of flock
            predators: list of all predator objects
        '''
        c_att = 15
        c_rep = 10
        c_avoid = 5

        l_att = 50
        l_rep = 200
        g=0.5
        

        align = np.zeros(2)
        att = np.zeros(2)
        rep = np.zeros(2)
        for boid in flock:
            if boid == self or distance(self.pos, boid.pos) > self.visual:
                    continue
            
            r = boid.pos - self.pos
            rnorm = np.linalg.norm(r)

            align += (1/rnorm**2)*(boid.velocity/np.linalg.norm(boid.velocity))
            att += np.exp(-rnorm/l_att)*(r/rnorm)
            rep += np.exp(-rnorm/l_rep)*(self.pos-boid.pos)/rnorm

        w=0.2
        R = 100
        avoid = np.zeros(2)
        for predator in predators:
            x_pi = (self.pos - predator.pos)/np.linalg.norm(self.pos-predator.pos)
            avoid += 1/(1+np.exp(w*(np.linalg.norm(predator.pos - self.pos)-R))) * x_pi

        gamma = 0.05
        self.velocity += align + c_att*att + c_rep*rep - c_avoid*avoid #- gamma*self.velocity


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
        side_length = 1 # how long triangle sides are
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
                maxSpeed = 2):
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
    visual_range = 200 #neighbourhood range
    flee_range = 200 #range at which boids will evade a predator
    urgent_range = 15 #range at which selfish evasive action is taken
    max_speed = 5
    max_acc = 5

    limits = [1000, 1000] #axes limits

    #generate boids and predators (with random positions and velocities)
    flock = [Boid((np.random.rand(2)*400+200), np.random.rand(2)*5) for _ in range(num_boids)] #generate boid objects

    predator_pos = np.array([600.5,650.5])
    predators = [Predator(predator_pos,  np.random.rand(2)*5) for _ in range(num_preds)]

    vd = np.array([1.0,1.5]) #pre selected steady state velocity

    #create figure and set axes limits
    figure = plt.figure()
    axes = plt.axes(xlim=[0,limits[0]], ylim = [0,limits[1]])

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

    #function called at each animation frame
    def animate(frame):
        for boid in flock:
            '''
            boid.flyTowardsCenter(visual_range, flock)
            boid.avoidOthers(flock)
            boid.matchVelocity(visual_range, flock)
            '''
            boid.flocking(flock, vd, predators)
            #alpha = boid.checkPredators(predators)
            boid.updatePosition()

            #update patches
            angle = np.arctan2(boid.velocity[1], boid.velocity[0])
            x1 = [boid.pos[0]-10*np.cos(angle-0.26),boid.pos[1]-10*np.sin(angle-0.26)]
            x2 = [boid.pos[0]-10*np.cos(angle+0.26),boid.pos[1]-10*np.sin(angle+0.26)]
            boid.tri = [boid.pos, x1, x2]
            boid.patch.set_xy(boid.tri)

        for predator in predators:
            predator.huntClosest(flock)
            predator.updatePosition()

            #update patches
            angle = np.arctan2(predator.velocity[1], predator.velocity[0])
            x1 = [predator.pos[0]-10*np.cos(angle-0.26),predator.pos[1]-10*np.sin(angle-0.26)]
            x2 = [predator.pos[0]-10*np.cos(angle+0.26),predator.pos[1]-10*np.sin(angle+0.26)]
            predator.tri = [predator.pos, x1, x2]
            predator.patch.set_xy(predator.tri)

    #run animation
    anim = animation.FuncAnimation(figure, animate,
                                    frames = 50, interval = 100)

    plt.show()
