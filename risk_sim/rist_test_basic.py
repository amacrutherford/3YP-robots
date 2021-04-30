import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib import patches
import random
import pandas as pd

class Board:

    def __init__(self,
                 risk_data,
                 iterations,
                 pzLoc,
                 pzRad,
                 accRad,
                 radarRad,
                 maxDeployRad,
                 minDeployRad,
                 noise,
                 k,
                 thresh,
                 minfP,
                 maxfP,
                 minfV,
                 maxfV):

        self.risk_data = risk_data
        self.iterations = iterations

        self.pzLoc = pzLoc
        self.pzRad = pzRad
        self.accRad = accRad
        self.maxDeployRad = maxDeployRad
        self.minDeployRad = minDeployRad
        self.radarRad = radarRad
        self.noise = 1 if noise else 0
        self.k = k
        self.thresh = thresh
        self.minfV = minfV
        self.maxfV = maxfV
        self.minfP = minfP
        self.maxfP = maxfP


        self.metThresh = {}
        self.ThreshDist = {}
        self.noThresh = []
        # testing
        self.ThreshRisk = {}


        self.flockin = []
        self.timesteps = []

    def normvec_topz(self, pos):
        vec = self.pzLoc - pos  # vector from flock to centre of pz
        return vec / np.linalg.norm(vec)  # normalise vector

    def risk_value(self, flock):
        dist = np.abs(np.linalg.norm(flock.pos - self.pzLoc))
        if dist < self.minDeployRad or dist > self.radarRad:  # ignore flock if within deployment range or out of range
            if dist < self.accRad:
                flock.isinpz = True
            return 0
        else:
            flock.isinpz = False

        velocity_norm = flock.vel / np.linalg.norm(flock.vel)
        direc_risk = (np.dot(velocity_norm, self.normvec_topz(flock.pos))+1)/2
        dist_risk = 1 - dist/(self.radarRad - self.accRad)
        return self.k*direc_risk + (1-self.k)*dist_risk

    def board_time_step(self, flocks):

        risk_values = [board.risk_value(flocks[i]) for i in range(len(flocks))]
        for i in range(len(risk_values)):  # update dictionary containing flocks above threshold
            if flocks[i].isinpz:
                if i in self.metThresh:
                    self.metThresh[i] = True
                elif i in self.noThresh:  # check if already in list of those not picked up by system
                    continue
                else:
                    self.noThresh.append(i)

            if (risk_values[i] > self.thresh) and (i not in self.metThresh.keys()) \
                    and (np.linalg.norm(flocks[i].pos - pzLoc) <= self.maxDeployRad):
                self.metThresh[i] = False
                self.ThreshDist[i] = np.abs(np.linalg.norm(flocks[i].pos - self.pzLoc))
                flocks[i].update_color()
                self.ThreshRisk[i] = risk_values[i]

        '''risk_values = [risk_values[i] if risk_values[i] > self.thresh else 0 for i in range(len(risk_values))]
        inpz = [flock.isinpz for flock in flocks]
        if any(inpz):
            print('flocks in pz', [i for i, x in enumerate(inpz) if x])
        print('risk values', risk_values)
        maxRiskId = risk_values.index(max(risk_values))
        if maxRiskId != self.maxRiskId and abs(risk_values[maxRiskId] - self.maxRiskVal) > 0.05:
            self.maxRiskId = maxRiskId
        if self.maxRiskId is not None:
            self.maxRiskVal = risk_values[self.maxRiskId]'''


    def init_flock_pos(self):
        r = np.random.uniform(self.minfP, self.maxfP)
        theta = np.random.uniform(0, 2 * np.pi)
        return self.pzLoc + np.array([np.cos(theta) * r, np.sin(theta) * r])

    def init_flock_vel(self):
        def rand_sign():
            return 1 if random.random() < 0.5 else -1
        return np.array([np.random.uniform(self.minfV, self.maxfV) * rand_sign(),
                         np.random.uniform(self.minfV, self.maxfV) * rand_sign()])

    def plot_board(self, ax):
        ax.add_patch(patches.Circle(self.pzLoc, radius=self.pzRad, ec='red', fill=False))  # pz circle
        ax.scatter(self.pzLoc[0], self.pzLoc[1], marker='+', c='r')  # pz center
        ax.add_patch(patches.Circle(self.pzLoc, radius=self.accRad, ec='purple', fill=False))  #accRad

        ax.add_patch(patches.Circle(self.pzLoc, radius=self.maxDeployRad, ec='gray', fill=False))  # maxDeployRad
        ax.add_patch(patches.Circle(self.pzLoc, radius=self.minDeployRad, ec='gray', fill=False))  # minDeployRad
        ax.add_patch(patches.Circle(self.pzLoc, radius=self.radarRad, ec='green', fill=False))  # radarRad
        loc = np.arange(9) * 250 - 500
        lab = (np.arange(9)*250 - 500)*0.01
        ax.set_xticks(loc)
        ax.set_yticks(loc)
        ax.set_xticklabels(lab)
        ax.set_yticklabels(lab)
        ax.set_xlabel('X (km)')
        ax.set_ylabel('Y (km)')

class Flock:

    def __init__(self,
                 board):

        self.pos = board.init_flock_pos()
        self.vel = board.init_flock_vel()
        self.noise = board.noise
        self.size = 15
        self.isinpz = False

    def update_pos(self):
        self.velocity_check()
        self.pos += self.vel

    def velocity_check(self):
        vmag = np.linalg.norm(self.vel) * (1 + self.noise * np.random.normal(0, 0.01))
        theta = np.arctan2(self.vel[1], self.vel[0]) * (1 + self.noise * np.random.normal(0, 0.02))
        self.vel = np.array([vmag * np.cos(theta), vmag * np.sin(theta)])

    def triangle_coords(self):  # coordinates for plotting of flock
        angle = np.arctan2(self.vel[1], self.vel[0])
        x1 = [self.pos[0] - self.size * np.cos(angle - 0.26), self.pos[1] - self.size * np.sin(angle - 0.26)]
        x2 = [self.pos[0] - self.size * np.cos(angle + 0.26), self.pos[1] - self.size * np.sin(angle + 0.26)]
        return [self.pos, x1, x2]

    def init_plot(self, ax):
        self.patch = patches.Polygon(self.triangle_coords(), closed=True, fc='b', ec='b')
        ax.add_patch(self.patch)

    def update_plot(self):
        self.patch.set_xy(self.triangle_coords())

    def update_color(self):
        self.patch.set_facecolor('red')
        self.patch.set_edgecolor('red')

def animate(frame):
    global hist_risk
    global flocks
    global j
    global board

    for i in range(len(flocks)):
        flocks[i].update_pos()
        flocks[i].update_plot()
        #update colour

        risk_values[i] = board.risk_value(flocks[i])
    board.board_time_step(flocks)
    if j % 100 == 0:
        print('iteration: ', j, '\nmet thresh', board.metThresh, '\n ThreshRisk', board.ThreshRisk, '\n no thresh', board.noThresh)
    if j == board.iterations:
        metThreshInPZ = sum(board.metThresh.values())/len(board.metThresh.values()) * 100
        print('*** Final results *** \n len metThresh:', len(board.metThresh.values()), '\n metThresh into pz:', metThreshInPZ,'% \n distance:', list(board.ThreshDist.values()),' \n len Others:', len(board.noThresh))
        test_data = {'Num_Met_Thresh': len(board.metThresh.values()), 'Distance':list(board.ThreshDist.values()),
                     'Entered_PZ': metThreshInPZ, 'Len_Others': len(board.noThresh), 'k': board.k, 'Thresh': board.thresh, 'maxDeployRad':board.maxDeployRad}
        board.risk_data = risk_data.append(test_data, ignore_index=True)
        #board.risk_data.to_json('risk_data.json')
    j += 1
    '''maxrisk = risk_values.index(max(risk_values))
    hist_risk.append([maxrisk, flocks[maxrisk].isinpz])
    print('maxrisk', maxrisk, 'risk value', risk_values[maxrisk])
    print('board maxrisk', board.maxRiskId, 'risk val', board.maxRiskVal)
    if risk_values[board.maxRiskId] < board.thresh:
        maxrisk_patch.center = board.pzLoc
    else:
        maxrisk_patch.center = flocks[board.maxRiskId].pos'''



if __name__ == '__main__':
    ''' board constants '''
    iterations = 600
    num_flock = 400
    radarRad = 800  # 10^2 m radar range
    pzLoc = np.array([500, 500])
    pzRad = 40  # 10^2 m
    accRad = 90
    maxDeployRad = 300  # 10^2 m
    minDeployRad = 150
    noise = True
    #k = 0.75
    k = 0.6
    #thresh = 0.895  # risk threshold
    thresh = 0.855

    minfP, maxfP = 400, 1150
    minfV, maxfV = 1, 3

    ''' plotting constants '''
    limits = [-500, 1500, -500, 1500]  # axes limits

    risk_data = pd.read_json('risk_data.json')

    board = Board(risk_data, iterations, pzLoc, pzRad, accRad, radarRad, maxDeployRad, minDeployRad, noise, k, thresh, minfP, maxfP, minfV, maxfV)

    flocks = [Flock(board) for _ in range(num_flock)]
    risk_values = [board.risk_value(flocks[i]) for i in range(len(flocks))]
    maxrisk = risk_values.index(max(risk_values))

    figure = plt.figure()
    ax1 = plt.axes(xlim=[limits[0], limits[1]], ylim = [limits[2], limits[3]])

    board.plot_board(ax1)
    for i in range(len(flocks)):
        flocks[i].init_plot(ax1)  # plot flocks

        # plot max risk
        '''if i == maxrisk:
            maxrisk_patch = patches.Circle(flocks[i].pos, radius=20, ec='red', fill=False)
            ax1.add_patch(maxrisk_patch)'''

    hist_risk = []
    j = 0

    # run animation
    anim = animation.FuncAnimation(figure, animate,
                                   frames=250, interval=50)


    #plt.show()
    f = r"/Users/alexrutherford/Documents/Ox Year 3/3yp/animations/risk_06.gif" 
    writergif = animation.PillowWriter(fps=15, bitrate=40) 
    anim.save(f, writer=writergif)