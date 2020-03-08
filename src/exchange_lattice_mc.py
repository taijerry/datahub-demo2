"""
Copyright 2019 Darren McAffee

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import numba
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors
import random


# need to use int, rather than uint, because finding dE uses negative numbers
spec = [
    ('nx', numba.int64),
    ('N', numba.int64),
    ('N2', numba.int64),
    ('NX', numba.int64),
    ('rg', numba.int16[:]),
    ('ix', numba.int16[:, :]),
    ('state', numba.int8[:, :]),
    ('states', numba.int8[:,:,:])
]


@numba.jitclass(spec)
class System:
    
    def __init__(self, Nx, X, steps, sampling):
        """
        Nx : number of tiles in the x direction
        X  : mole fraction of blue tiles
        """
        nx = Nx
        self.nx = nx
        N = nx*nx
        self.N = N
        self.N2 = N//2
        self.NX = int(N * X)
        self.rg = np.arange(N).astype(np.int16)
        self.ix = np.zeros((N, 2)).astype(np.int16)
        for i in range(nx):
            for j in range(nx):
                self.ix[i*nx + j] = np.array([i, j], dtype=np.int16)
        self.state = np.zeros(self.N, dtype=np.int8).reshape(nx, nx)
        x = self.sample_ix(self.NX)
        for i in range(self.NX):
            self.state[x[i, 0], x[i, 1]] = 1
        self.states = np.zeros((steps//sampling, nx, nx), dtype=np.int8)
        self.states[0] = self.state
    
    def sample_pairs(self, n):
        # sample n pairs, returns array of (n*2, 2)
        assert 2*n <= self.N
        st = set(range(self.N))
        sample = np.zeros((2*n,2)).astype(np.int16)
        i = 0
        while len(st) > 0 and i < 2*n:
            rn = np.random.randint(self.N)
            if rn in st:
                rsite = self.ix[rn]
                nset = [0,1,2,3]
                while len(nset) > 0:
                    rnbr = random.randint(0, 3)
                    if rnbr in nset:
                        nset.remove(rnbr)
                    else:
                        continue
                    rnsite = rsite.copy()
                    # lower left [0,0], top left [m, 0], top right [m, n], lower right [0, n]
                    # if self.state[rsite[0], rsite[1]] > 0:
                    #     rnbr = 0
                    if rnbr == 0:  # down
                        rnsite[1] -= 1
                    elif rnbr == 1:  # left
                        rnsite[0] -= 1
                    elif rnbr == 2:  # up
                        rnsite[1] += 1
                    elif rnbr == 3:  # right
                        rnsite[0] += 1
                    rnsite2 = np.array([rnsite[0]%self.nx, rnsite[1]%self.nx], dtype=np.int16)
                    rn2 = rnsite2[0]*self.nx + rnsite2[1]
                    if rn2 in st:
                        # if self.state[rnsite2[0], rnsite2[1]] > 0:
                        #     continue
                        # elif self.state[rsite[0], rsite[1]] > 0:
                        #     print(rsite, rn, rnbr, rnsite, rnsite2, rn2)
                        sample[i] = rsite
                        sample[i+1] = rnsite2
                        st.remove(rn)
                        st.remove(rn2)
                        i += 2
                        break
                else:
                    # executes when while becomes false
                    st.remove(rn)  # can't exchange with any neighbors
        return sample

    def sample_ix(self, n):
        return self.ix[np.random.choice(self.rg, size=n, replace=False)]


@numba.jit(nopython=True)
def isclose(a, b, rtol=1e-5, atol=1e-8):
    """
    Helps determine if arrays of floats are "the same". This is a helper function 
    since numba does not yet support numpy.isclose()
    """
    return np.abs(a - b) <= (atol + rtol * np.abs(b))


@numba.jit(nopython=True)
def nneighbors(state, c, s, nx):
    xr = (c[0]+1)%nx
    right = state[xr, c[1]]
    xl = (c[0]-1)%nx
    left = state[xl, c[1]]
    yu = (c[1]+1)%nx
    up = state[c[0], yu]
    yd = (c[1]-1)%nx
    down = state[c[0], yd]
    ns = right + left + up + down
    if s == 0:
        return 4 - ns
    else:
        return ns


@numba.jit(nopython=True)
def are_neighbors(c0, c1, nx):
    m = np.abs(c0-c1)
    if isclose(m, 1.).any():
        return True
    elif isclose(m, nx).any():
        if isclose(m[0], 0) or isclose(m[1], 0):
            return True
    return False


@numba.jit(nopython=True)
def MCswap(system, E, T, Ew, ixs=None, report=False, neighbors=True):
    if ixs is None:
        N = system.N2
        N -= N%2
        if neighbors:
            sx = system.sample_pairs(N)
        else:
            sx = system.sample_ix(2*N)
    else:
        sx = ixs
        N = len(ixs)//2
    # ll = np.where(system.state > 0)
    # print(ll)
    # print(dcs[sx == ll])
    for i in range(N):
        ixs = sx[2*i:2*i+2]
        s0, s1 =  system.state[ixs[0,0], ixs[0,1]], system.state[ixs[1,0], ixs[1,1]]
        if s0 != s1:
            # get number of same initial neighbors
            c00i = nneighbors(system.state, ixs[0], s0, system.nx)
            c11i = nneighbors(system.state, ixs[1], s1, system.nx)
            # final state
            c00f = nneighbors(system.state, ixs[0], s1, system.nx)
            c11f = nneighbors(system.state, ixs[1], s0, system.nx)
            if neighbors or are_neighbors(ixs[0], ixs[1], system.nx):
                ab_i = 7 - c11i - c00i
                ab_f = 7 - c11f - c00f + 2 # each neighbor count will count a state next to itself
            else:
                ab_i = 8 - c11i - c00i
                ab_f = 8 - c11f - c00f
            nab = ab_f - ab_i
            # dEx = final - initial
            # ∆U = ∆mab * (exchange paramter / z) see Molecular Driving Forces, 2nd, (15.9)
            # Here, E = (wab - (waa + wbb)/2). So X = z E/T (15.13) in dimensionless units
            # And criticality is achieved at X=2, so when 
            # ***E = T/2**** (using z=4).
            dEx = nab * E
            # get "wall" energy from attraction to left wall.
            dEw = (s1*ixs[0,1]+s0*ixs[1,1] - s1*ixs[1,1]-s0*ixs[0,1])*Ew  # only 1's are attracted to wall
            dE = dEx + dEw
            rn = np.random.rand()
            if dE <= 0 or rn < np.exp(-dE/T):
                if dE > 0 and report:
                    print("∆n(ab): ", nab, "  ∆E: ", dE, "  rand: ", rn, "  exp(-dE/T): ", np.exp(-dE/T))
                # more similar neighbors gained than lost, or sufficient energy acquired
                system.state[ixs[0,0], ixs[0,1]], system.state[ixs[1,0], ixs[1,1]] = s1, s0


@numba.jit(nopython=True)
def runMC(steps=2000, sampling=2, nx=10, E=3.0, T=5.0, Ew=0.0, X=3, neighbors=True, report=False):
    # initialize
    sys = System(nx, X, steps, sampling)
    
    # run
    for i in range(1, steps):
        MCswap(sys, E, T, Ew, neighbors=neighbors, report=report)
        if i % sampling == 0:
            sys.states[i//sampling] = sys.state
    return sys

cmap = colors.ListedColormap(['goldenrod', 'royalblue'])

def showMC(states, ticks=False, interval=None):
    bounds = [0,1]
    nx = states.shape[1]
    # norm = colors.BoundaryNorm(bounds, cmap.N)  # use with ax.imshow(..., norm=norm) remaps to [0,1]
    
    fig, ax = plt.subplots(figsize=(6,6))
    shape = states.shape
    if np.ndim(states) > 2:
        extent = (0, shape[2], shape[1], 0)
    else:
        extent = (0, shape[1], shape[0], 0)
    
    if np.ndim(states) == 2:
        imax = ax.imshow(states, cmap=cmap, extent=extent, origin='upper')
    else:
        imax = ax.imshow(states[0], cmap=cmap, extent=extent, origin='upper')
    
    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    if ticks:
        ax.set_xticks(np.arange(0, extent[1]+1, 1))
        ax.set_yticks(np.arange(0, extent[2]+1, 1))
    else:
        ax.axis('off')
    
    n = len(states)
    
    def advance(fn):
        imax.set_data(states[fn % n])
        ax.set_title(fn % n)
    if np.ndim(states) == 3:
        if interval is None:
            it = 5e3//n
        else:
            it = interval
        fa = animation.FuncAnimation(fig, advance, interval=it, save_count=len(states))
    else:
        fa = ax
    return fa


def saveMC(states, ticks=False, interval=None):
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    fa = showMC(states, ticks, interval)
    fa.save('mc.mp4', writer=writer)

