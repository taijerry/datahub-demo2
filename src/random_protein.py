"""
This module simulates a hardshell 2D gas. With relatively simple code we can emulate elastic collisions between
particles in a box with no detectable energy leakage.

The parameters of the simulation must be tuned well. Typically don't want more than 1/20th of the ball radius to 
overlap during a collision.

Ternary and higher-order collisions are not computed perfectly accurately, but energy is conserved.

Typical usage is:
    * setup v, x, m, dt, diam, Lx variables
    * run hardshell function on these variables
    * run animate on these variables


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
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.seterr(all='raise')  # floating point errors are raised

from scipy.stats import multivariate_normal, rv_continuous

beta = multivariate_normal(mean=[-120, 120], cov=[500, 400])
alpha = multivariate_normal(mean=[-50, -60], cov=[100, 200])

def plot_anglePD(pd):
    x, y = np.mgrid[-180:180:1, -180:180:1]
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x; pos[:, :, 1] = y
    fig, ax = plt.subplots(figsize=(3,3))
    plt.contour(x, y, pd.pdf(pos))
    plt.axis('equal')
    plt.xlabel('$\phi$')
    plt.ylabel('$\psi$')
    plt.tight_layout()


def psiomegaphi(n):
    # n, ca, c, n, ca, c...
    # psi encountered first (eclipsing N's)
    # omega encountered second (eclipsing Ca's)
    # phi...
    r = np.random.rand(n)
    a = r < 0.6
    na = a.sum()
    nb = n - na
    al = alpha.rvs(size=na).reshape(-1, 2)
    be = beta.rvs(size=nb).reshape(-1, 2)
    ags = np.concatenate((al, be))
    o = ags[np.random.choice(len(ags), len(ags), replace=False)]
    o = np.append(o, np.ones((len(o), 1))*180, axis=1)
    o = o[:, [1, 2, 0]]
    return o


@numba.jit(nopython=True)
def cross3(vec1, vec2):
    """ Calculate the cross product of two 3d vectors. """
    result = np.ones(3)
    a1, a2, a3 = vec1[0], vec1[1], vec1[2]
    b1, b2, b3 = vec2[0], vec2[1], vec2[2]
    result[0] = a2 * b3 - a3 * b2
    result[1] = a3 * b1 - a1 * b3
    result[2] = a1 * b2 - a2 * b1
    return result


@numba.jit(nopython=True)
def dot(vec1, vec2):
    """ Calculate the dot product of two 3d vectors. """
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]

@numba.jit(nopython=True)
def angle2vec(vec, theta, axis):
    # Rodrigues-Gibbs formulation
    # https://onlinelibrary.wiley.com/doi/epdf/10.1002/jcc.20237
    rad = (theta / 180) * np.pi
    c1 = vec * np.cos(rad)
    c2 = cross3(axis, vec)*np.sin(rad)
    c3 = dot(vec, axis)*axis*(1-np.cos(rad))
    return c1 + c2 + c3

# # test cross
# x = np.array([1, 0, 0])
# y = np.array([0, 1, 0])
# cross3(x, y)

# test angle2vec
# x = np.array([1, 0, 0])
# y = np.array([0, 1, 0])
# angle2vec(x, 90, y) # should be [0, 0, -1]

@numba.jit(nopython=True)
def angles2vectors(angles):
    ags = angles.ravel()
    o = np.zeros((len(ags)+2, 3))
    bondangle = 180-109.5
    o[0] = np.array([1, 0, 0])
    o[1] = angle2vec(np.array([0,1,0]), bondangle, o[0])
    for i in range(2, len(ags)+2):
        d0 = o[i-1]
        axis0 = cross3(o[i-2], d0)
        axis0 = axis0/np.linalg.norm(axis0)
        d1 = angle2vec(d0, bondangle,  axis0)
        d2 = angle2vec(d1, ags[i-2], d0)
        d2 = d2/np.linalg.norm(d2)
        o[i] = d2
    return o

# def set_axes_equal(ax):
#     '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
#     cubes as cubes, etc..  This is one possible solution to Matplotlib's
#     ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    
#     Input
#       ax: a matplotlib axis, e.g., as output from plt.gca().
#     '''
    
#     limits = np.array([
#         ax.get_xlim3d(),
#         ax.get_ylim3d(),
#         ax.get_zlim3d(),
#     ])
    
#     origin = np.mean(limits, axis=1)
#     radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
#     set_axes_radius(ax, origin, radius)


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



def plot3d(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = points.T
    ax.plot(x, y, z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    set_axes_equal(ax)
    return ax

def random_alpha(n=20):
    ag = alpha.rvs(size=n)
    ag = np.append(ag, np.ones((len(ag), 1))*180, axis=1)
    ag = ag[:, [1, 2, 0]]
    vecs = angles2vectors(ag)
    points = vecs.cumsum(axis=0)
    plot3d(points)

@numba.jit(nopython=True)
def blockcumsum(arr, size=50):
    N = len(arr)
    o = np.zeros((arr.shape[0]//size-1, arr.shape[1]))
    for i in range(1, len(o)+1):
        o[i-1] = arr[size*(i-1):size*i].sum(axis=0)
    return o

def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])
