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
                visual = 15,
                flee = 10,
                urgent = 5,
                maxSpeed = 1,
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

    def flocking(self, flock, predators):
        ''' Implements the cohesion condition - steer to move towards the average position of flockmates
        Args:
            flock: list of all boid objects
        '''
        #flocking weights -- adjust velocity by this %
        centering = 0.0015
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

        #predator avoidance
        predator_weight = 0.1 #adjust velocity by this %
        for predator in predators:
            if distance(self.pos, predator.pos) <= self.flee:
                self.velocity += (self.pos - predator.pos) * predator_weight  

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
                maxSpeed = 1):
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
        x1 = [self.pos[0]-side_length*np.cos(angle-angle_adjustment),self.pos[1]-side_length*np.sin(angle-angle_adjustment)]
        x2 = [self.pos[0]-side_length*np.cos(angle+angle_adjustment),self.pos[1]-side_length*np.sin(angle+angle_adjustment)]
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

    #generate boids and predators (with random positions and velocities)
    flock = [Boid((np.random.rand(2)*400+200), np.random.rand(2)*5) for _ in range(num_boids)] #generate boid objects
    #flock = [Boid(np.array([675.5, 675.5]), np.random.rand(2)*5)]

    predator_pos = np.array([600.5,650.5])
    predators = [Predator(predator_pos,  np.random.rand(2)*5) for _ in range(num_preds)]

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

        #plot flee radius --> probably temporary
        predator.radius = patches.Circle(predator.pos, radius=flee_range, ec='red', fill=False)
        axes.add_patch(predator.radius)

    #function called at each animation frame
    def animate(frame):
        for boid in flock:
            boid.flocking(flock, predators)
            boid.updatePosition()

            #update patches
            boid.tri_coords()
            boid.patch.set_xy(boid.tri)

        for predator in predators:
            predator.huntClosest(flock)
            predator.updatePosition()

            #update patches
            predator.tri_coords()
            predator.patch.set_xy(predator.tri)
            predator.radius.center = predator.pos

    #run animation
    anim = animation.FuncAnimation(figure, animate,
                                    frames = 50, interval = 50)

    plt.show()