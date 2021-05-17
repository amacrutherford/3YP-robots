''' Boids simulation for 3YP Project
Author: Alex Rutherford
'''
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib import patches 

#yet to add the angle for neighbourhood, just distance based
#need to add a minimum speed check (birds cannot fly with zero velocity)

def distance(x, y):
    ''' Euclidean distance between two points '''
    return np.linalg.norm(np.array(x) - np.array(y))

class Boid():
    ''' class to hold information for each boid/bird
    Args:
        pos = initial position of boid
        velocity = initial velocity of boid
    '''

    def __init__(self,
                pos,
                velocity):
        
        self.pos = pos
        self.velocity = velocity

        #for plotting
        self.patch = None #patches.Polygon (triangle) object associated with the boid
        self.tri = None #coordiates of the triangle

    def flyTowardsCenter(self, visual_range, flock):
        ''' Implements the cohesion condition - steer to move towards the average position of flockmates
        Args:
            visual_range: sets the neighorhood distance
            flock: list of all boid objects
        '''
        centering_weight = 0.005 #adjust velocity by this %

        num_neighbours = 0
        centre = np.zeros(2)
        for boid in flock:
            if boid == self:
                continue
            if distance(self.pos, boid.pos) < visual_range:
                num_neighbours += 1
                centre += boid.pos
        
        if num_neighbours != 0:
            centre = centre / num_neighbours
            self.velocity += (centre - self.pos) * centering_weight

    def avoidOthers(self, flock):
        ''' Implements the seperation condition - steer to avoid crowding local flockmates
        Args:
            flock: list of all boid objects

        Should this vary with distance (i.e. get stronger the closer the bird is?)
        '''
        min_distance = 30 #the distance to stay away from other birds
        avoid_weight = 0.05 #adjust velocity by this %

        move = np.zeros(2)
        for boid in flock:
            if boid == self:
                continue
            if distance(self.pos, boid.pos) < min_distance:
                move += self.pos - boid.pos

        self.velocity += move * avoid_weight

    def matchVelocity(self, visual_range, flock):
        ''' Implements the alighnment condition - steer towards the average heading of local flockmates
        Args:
            visual_range: set the neighorhood distance
            flock: list of all boid objects
        '''
        matching_weight = 0.05 #adjust velocity by this %

        avg_vel = np.zeros(2)
        num_neighbours = 0

        for boid in flock:
            if boid == self:
                continue
            if distance(self.pos, boid.pos) < visual_range:
                num_neighbours += 1
                avg_vel += boid.velocity

        if num_neighbours != 0:
            avg_vel = avg_vel / num_neighbours
            self.velocity += (avg_vel - self.velocity) * matching_weight

    def checkPredators(self, flee_range, predators):
        ''' Implements a basic sense of predator avoidance
        Args:
            flee_range: range at which boid will free from predator
            predators: list of all predators
        '''
        predator_weight = 0.75
        move = 0
        for predator in predators:
            if distance(self.pos, predator.pos) < flee_range:
                move += self.pos - predator.pos
        self.velocity += move * predator_weight

    def updatePosition(self):
        ''' Updates the boids position using velocity
        Implements a maximum speed check (birds can only fly so fast)
        '''
        speed_limit = 5
        speed = np.linalg.norm(self.velocity)
        if speed > speed_limit:
            self.velocity = (self.velocity/speed) * speed_limit

        self.pos += self.velocity

class Predator():

    def __init__(self,
                pos,
                velocity=np.zeros(2)):
        self.pos = pos
        self.velocity = velocity

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
        ''' Updates the boids position using velocity
        Implements a maximum speed check (birds can only fly so fast)
        '''
        speed_limit = 7
        speed = np.linalg.norm(self.velocity)
        if speed > speed_limit:
            self.velocity = (self.velocity/speed) * speed_limit
        self.pos += self.velocity


#CONSTANTS
num_boids = 40 #number of boids
num_preds = 1 #number of predators
visual_range = 200 #neighbourhood range
flee_range = 200 #range at which boids will evade a predator

limits = [1000, 1000] #axes limits

flock = [Boid((np.random.rand(2)*400+200), np.random.rand(2)*5) for _ in range(num_boids)] #generate boid objects

predator_pos = np.array([700.5,700.5])
predators = [Predator(predator_pos) for _ in range(num_preds)]

#create figure and set axis limits
figure = plt.figure()
axes = plt.axes(xlim=[0,limits[0]], ylim = [0,limits[1]])

#plot initial positions of flock
for boid in flock:
    angle = np.arctan2(boid.velocity[1], boid.velocity[0])
    x1 = [boid.pos[0]-5*np.cos(angle-0.26),boid.pos[1]-5*np.sin(angle-0.26)]
    x2 = [boid.pos[0]-5*np.cos(angle+0.26),boid.pos[1]-5*np.sin(angle+0.26)]
    boid.tri = [boid.pos, x1, x2]
    boid.patch = patches.Polygon(boid.tri, closed=True, fc='b', ec='b')
    axes.add_patch(boid.patch)

#plot initial positions of predators
for predator in predators:
    angle = np.arctan2(predator.velocity[1], predator.velocity[0])
    x1 = [predator.pos[0]-10*np.cos(angle-0.26),predator.pos[1]-10*np.sin(angle-0.26)]
    x2 = [predator.pos[0]-10*np.cos(angle+0.26),predator.pos[1]-10*np.sin(angle+0.26)]
    predator.tri = [predator.pos, x1, x2]
    predator.patch = patches.Polygon(predator.tri, closed=True, fc='r', ec='r')
    axes.add_patch(predator.patch)

    #plot flee radius
    predator.radius = patches.Circle(predator.pos, radius=flee_range, ec='red', fill=False)
    axes.add_patch(predator.radius)

#function called at each animation frame
def animate(frame):
    for boid in flock:
        boid.flyTowardsCenter(visual_range, flock)
        boid.avoidOthers(flock)
        boid.matchVelocity(visual_range, flock)
        boid.checkPredators(flee_range, predators)
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
        predator.radius.center = predator.pos
        


        


#run animation
anim = animation.FuncAnimation(figure, animate,
                                frames = 50, interval = 50)

plt.show()
