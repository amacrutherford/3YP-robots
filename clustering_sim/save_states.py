import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib import patches
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering

#boids and positions lists must be in same order

cmap = {0:'red', 1:'blue', 2:'green', 3:'orange', 4:'cyan', 5:'black'}

def distance(x, y):
    ''' Euclidean distance between two points '''
    return np.linalg.norm(np.array(x) - np.array(y))

def normvec_topz(pos, pz_loc):
    vec = pz_loc - pos #vector from flock to centre of pz
    return vec / np.linalg.norm(vec) #normalise vector

def risk_value(pos, velocity, pz_loc, range_threshold):
    dist = np.abs(np.linalg.norm(pos - pz_loc))
    if dist > range_threshold:
        return 0
    else:
        velocity_norm = velocity / np.linalg.norm(velocity)
        direc_risk = (np.dot(velocity_norm, normvec_topz(pos, pz_loc))+1)/2
        dist_risk = 1 - dist/range_threshold
        return (direc_risk + dist_risk)/2

def return_clusters(states):
    ''' Groups the states into clusters, determines the number of clusters from the largest difference in the Dendrogram
    Args:
        states = the states to be grouped into clusters
    Returns:
        clusters.labels_ = list containing a label for each state mapping it to a cluster
    '''
    link_meth = 'single'
    if len(states) < 2:
        return [0 for state in states]
    Z = hierarchy.linkage(states, method=link_meth)
    differences = [Z[i,2] - Z[i-1,2] for i in range(1,len(Z[:,2]))]
    if differences == []:
        num_clust = 1
    else:
        num_clust = len(Z[:,2]) - np.argmax(differences)
    #print('num clust', num_clust)

    ''' no max cluster
    if num_clust > self.max_clust:
        num_clust = self.max_clust '''

    cluster = AgglomerativeClustering(n_clusters=num_clust, affinity='euclidean', linkage=link_meth)  
    cluster.fit_predict(states)
    return cluster.labels_

def gen_boids(num_flocks):
    offset = [np.random.randint(limits[0]-300), np.random.randint(limits[1]-300)]
    positions = np.array([ np.random.rand(2)*50 + offset for i in range(np.random.randint(3, 10))])
    for _ in range(num_flocks-1):
        offset = [np.random.randint(limits[0]), np.random.randint(limits[1])]
        pos = np.array([ np.random.rand(2)*50 + offset for i in range(np.random.randint(3, 10))])
        positions = np.concatenate((positions, pos), axis=0)
    
    return [Boid(positions[i], np.random.rand(2)*2) for i in range(len(positions))] #generate boid objects


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
        
        self.flock = None

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

'''
class pz():

    def __init__(self,
                pz_loc,
                pz_rad,
                radar_range):
        
        self.pz_loc = pz_loc
        self.pz_rad = pz_rad
        self.radar_range = radar_range
'''

#CONSTANTS
num_flocks = 7 #number of boids
visual_range = 200 #neighbourhood range
flee_range = 200 #range at which boids will evade a predator

range_threshold = 750
pz_loc = np.array([1000,1000])
pz_rad = 100

boid_data = pd.read_json('boid_data.json')
j = 0

limits = [2000, 2000] #axes limits

boids = gen_boids(num_flocks)

#create figure and set axis limits
figure = plt.figure()
axes = plt.axes(xlim=[-100,limits[0]+100], ylim = [-100,limits[1]+100])
axes.set_xlabel('X (m)')
axes.set_ylabel('Y (m)')

#plot initial positions of flock
flock_labels = return_clusters([boid.pos for boid in boids])
#print('labels', flock_labels)
for boid in boids:
    angle = np.arctan2(boid.velocity[1], boid.velocity[0])
    x1 = [boid.pos[0]-5*np.cos(angle-0.26),boid.pos[1]-5*np.sin(angle-0.26)]
    x2 = [boid.pos[0]-5*np.cos(angle+0.26),boid.pos[1]-5*np.sin(angle+0.26)]
    boid.tri = [boid.pos, x1, x2]
    boid.patch = patches.Polygon(boid.tri, closed=True, fc='b', ec='b')
    axes.add_patch(boid.patch)

#plot pz
pz_patch = patches.Circle(pz_loc, radius=pz_rad, ec='red', fill=False)
axes.add_patch(pz_patch)
plt.plot(pz_loc[0], pz_loc[1], marker='+', c='r')

#plot range
axes.add_patch(patches.Circle(pz_loc, radius=range_threshold, ec='green', fill=False))

#risk_id = patches.Circle(pz_loc, radius = 20, ec='black', fill=False)
#axes.add_patch(risk_id)

def flock_metrics(boids, flock_labels):
    #print('len boids', len(boids), 'len flock_labels', len(flock_labels))

    flocks = []
    fpos = []
    fvel = []
    for label in range(max(flock_labels)+1):
        flock = [boids[i] for i in range(len(boids)) if flock_labels[i] == label]
        fpos.append(np.sum([boid.pos for boid in flock], axis=0)/len(flock))
        fvel.append(np.sum([boid.velocity for boid in flock], axis=0)/len(flock))
        flocks.append(flock)
    
    return flocks, fpos, fvel



#initalise risk values
flock_labels = return_clusters([boid.pos for boid in boids])
flocks, fpos, fvel = flock_metrics(boids, flock_labels)
risk_values = [risk_value(fpos[i], fvel[i], pz_loc, range_threshold) for i in range(len(flocks))] 
maxrisk_patch = patches.Circle(fpos[risk_values.index(max(risk_values))], radius=30, ec='red', fill=False)
axes.add_patch(maxrisk_patch)


#function called at each animation frame
def animate(frame):
    global j
    global boid_data
    #determine flocks and risk
    #positions = [boid.pos for boid in boids]
    flock_labels = return_clusters([boid.pos for boid in boids])
    flocks, fpos, fvel = flock_metrics(boids, flock_labels)
    #print('fpos', fpos)
    #print('fvel', fvel)
    risk_values = [risk_value(fpos[i], fvel[i], pz_loc, range_threshold) for i in range(len(flocks))]
    if max(risk_values) > 0:
        #print(risk_values.index(max(risk_values)))
        #print('new id centre',  fpos[risk_values.index(max(risk_values))])
        new_pos = fpos[risk_values.index(max(risk_values))]
        #risk_id.centre = new_pos
        #axes.add_patch(patches.Circle(new_pos, radius=20))
        plt.scatter(new_pos[0], new_pos[1],c='white')
        maxrisk_patch.center = new_pos
    for boid in boids:
        boid.flyTowardsCenter(visual_range, boids)
        boid.avoidOthers(boids)
        boid.matchVelocity(visual_range, boids)
        #boid.checkPredators(flee_range, predators)
        boid.updatePosition()

        #update patches
        angle = np.arctan2(boid.velocity[1], boid.velocity[0])
        size = 15
        x1 = [boid.pos[0]-size*np.cos(angle-0.26),boid.pos[1]-size*np.sin(angle-0.26)]
        x2 = [boid.pos[0]-size*np.cos(angle+0.26),boid.pos[1]-size*np.sin(angle+0.26)]
        boid.tri = [boid.pos, x1, x2]
        boid.patch.set_xy(boid.tri)

        boid.patch.set_color(cmap[flock_labels[boids.index(boid)]])

    if j==50:
        print('500000')
        plt.savefig('boids_at_50.png')  # save figure
        b_data = {'NumClust': max(flock_labels)+1 , 'States': [boid.pos for boid in boids]}
        boid_data = boid_data.append(b_data, ignore_index=True)
        boid_data.to_json('boid_data.json')

    j += 1


#run animation
anim = animation.FuncAnimation(figure, animate,
                                frames = 150, interval = 50)


'''f = r"/Users/alexrutherford/Documents/Ox Year 3/3yp/animations/risk_w_cluster.gif" 
writergif = animation.PillowWriter(fps=30) 
anim.save(f, writer=writergif)'''
plt.show()