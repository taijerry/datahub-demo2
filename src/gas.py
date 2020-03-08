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

import numba, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PatchCollection

np.seterr(all='raise')  # floating point errors are raised


# Main functions to create trajectory for 2D gas
# While this is not "bullet proof" (some pathologies may arise for large time steps),
# we can get fairly robust performance with 4 simple functions.


@numba.jit(nopython=True)
def hardshell(v, x, m, r, dt=1e-3, lims=(12, 12)):
    """
    Takes a velocity array and a position array that are 0's except for v[0] and x[0].
    Based on initial velocities updates positions and velocities based on hardshell 
    elastic colliions.
    
    v: initialized velocity array of shape (steps, nparticles, 2)
    x: initialized coordinates array of shape (steps, nparticles, 2)
    m: 1d array of masses
    r: 1d array of radii
    dt: timestep. For X's ~10 and V's ~1, dt should be ~1e-2 or less
    lims: box size, (extent in x, extent in y)
    
    It is important to run this on recently initialized v and x. Recycling arrays will
    cause errors.
    """
    s, n, p = x.shape
    for h in range(1, s):
        for i in range(n): # for each particle
            # update position based off initial velocity
            x[h, i] = x[h-1, i] + v[h-1, i] * dt
        # for detecting collisions
        d = np.zeros((n, n))
        for i in range(n):
            v[h, i] = v[h-1, i]
            for j in range(i+1, n):
                d[i, j] = np.sqrt( ((x[h, i] - x[h, j])**2).sum() )
        # this approach may seem to repeat a for loop, but the structure is more extensible for adding forces
        # resolve all collisions simultaneously
        for i in range(n):
            for j in range(i+1, n):
                if d[i, j] < r[i] + r[j]:
                    if h > 1:
                        mxpre = np.sqrt(((x[h-1, i] - x[h-1, j])**2).sum())
                        if mxpre > r[i] + r[j]:
                            # first time they've encountered.
                            vp = collision2(v[h, i], v[h, j], x[h, i], x[h, j], m[i], m[j])
                            v[h, i] = vp[0]
                            v[h, j] = vp[1]
            # handle walls
            pw = pastwall(x[h, i], lims, r[i])
            if pw.any():
                # hardshell collision. no forces until it escapes (in elif)
                if h > 1:
                    pw2 = pastwall(x[h-1, i], lims, r[i])
                    if not isclose(pw2, pw).all():
                        # first time hitting this wall
                        invert = -(pw**2)
                        v[h, i] = v[h, i] + 2*invert*v[h, i]
            # ensure that after resolving collisions and walls that velocities point the right way
            v[h, i] = velocity_compatible(x[h, i], v[h, i], lims)


@numba.jit(nopython=True)
def collision2(v1, v2, x1, x2, m1, m2):
    """
    Calculate an elastic collision between two masses. If one particle is coming back from
    the wall it is treated as a wall (i.e. infinite inertia), which allows it to recover back
    within the boundaries.
    """
    d = m1 + m2
    dv = v1-v2
    dx = x1-x2
    vps = np.zeros((2,2))
    # make an elastic collision
    vps[0] = v1 - 2*m2/d * (dv * dx).sum() / (dx * dx).sum() * dx
    dv *= -1
    dx *= -1
    vps[1] = v2 - 2*m1/d * (dv * dx).sum() / (dx * dx).sum() * dx
    return vps


@numba.jit(nopython=True)
def pastwall(x, lims, r):
    """
    Returns [1,0] if right wall is exceeded, [-1,0] if left wall is exceeded, etc.
    """
    o = np.zeros(2)
    if x[0] < r:
        o[0] = -1
    elif x[0] > lims[0] - r:
        o[0] = 1
    if x[1] < r:
        o[1] = -1
    elif x[1] > lims[1] - r:
        o[1] = 1
    return o


@numba.jit(nopython=True)
def velocity_compatible(x, v, lims):
    """
    Higher order collisions in a time-step may continue to change velocities in ways that
    are incompatible with the walls. Given a position, this ensures the particle is 
    oriented correctly.
    """
    vo = v.copy()
    if v[0] < 0 and x[0] < 0:
        vo[0] = -1*v[0]
    if v[0] > 0 and x[1] > lims[0]:
        vo[0] = -1*v[0]
    if v[1] < 0 and x[1] < 0:
        vo[1] = -1*v[1]
    if v[1] > 0 and x[1] > lims[1]:
        vo[1] = -1*v[1]
    return vo


@numba.jit(nopython=True)
def isclose(a, b, rtol=1e-5, atol=1e-8):
    """
    Helps determine if arrays of floats are "the same". This is a helper function 
    since numba does not yet support numpy.isclose()
    """
    return np.abs(a - b) <= (atol + rtol * np.abs(b))


# Visualization and validation functions
def animate(x, rs, label=False, lims=(12, 12), cols=[]):
    """
    Takes an array of positions with shape = (steps, particle, coordinates) and spot size (r)
    and animates their trajectory.
    """
    if not plt.get_fignums():
        fig, ax = plt.subplots(figsize=(5,5))
    else:
        ax = plt.gca()
        fig = plt.gcf()
    x = np.squeeze(x)  # if there is one particle
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    # use plt.Circle over ax.scatter since Circle size is in data coordinates, but
    # scatter marker sizes are in Figure coordinates.
    c = PatchCollection([plt.Circle((0, 0), r) for r in rs])
    c.set_offset_position('data')
    c.set_offsets(x[0])
    if not cols and x.shape[1] > 10:
        cols = ['r' if i < lims[0]/2 else 'b' for i in x[0, :, 0]]
    elif cols:
        pass
    else:
        cols = ['r' for i in x[0, :]]
    c.set_color(cols)
    ax.add_collection(c)
    def advance(fn):
        step = fn % len(x)
        c.set_offsets(x[step])
        ax.set_title(step)
        for l, xy in zip(ls, x[step]):
            l.set_x(xy[0])
            l.set_y(xy[1])
    if label:
        ls = [plt.text(*xy, i) for i, xy in enumerate(x[0])]
    else:
        ls = []
    ax.set_xlim((0, lims[0]))
    ax.set_ylim((0, lims[0]))
    ax.add_collection(c)
    fa = animation.FuncAnimation(fig, advance, interval=5e3//len(x))
    plt.show()
    return fa


def num_in_box(x, lim, r):
    s = (x < lim + r) & (x > 0 - r)
    s = s.all(axis=2)
    return s.sum(axis=1)


@numba.jit(nopython=True)
def hardshell_semiperm(v, x, m, r, k, wallx=6.0, dt=1e-3, lims=(12., 12.)):
    """
    Similar to hardshell, but introduces an inner wall into the box at 'wallx'.
    Particles with indices less than 'k' will collide with wallx, but the rest of the particles
    will only be affected by the walls at lims.
    
    Takes a velocity array and a position array that are 0's except for v[0] and x[0].
    Based on initial velocities updates positions and velocities based on hardshell 
    elastic colliions.
    
    v: initialized velocity array of shape (steps, nparticles, 2)
    x: initialized coordinates array of shape (steps, nparticles, 2)
    m: 1d array of masses
    r: 1d array of radii
    k: particles with indices lower than this will be subject to wallx
    wallx: where a "new wall" is for particles with index less than k
    dt: timestep. For X's ~10 and V's ~1, dt should be ~1e-2 or less
    lims: box size, (extent in x, extent in y)
    
    It is important to run this on recently initialized v and x. Recycling arrays will
    cause errors.
    """
    s, n, p = x.shape
    for h in range(1, s):
        for i in range(n): # for each particle
            # update position based off initial velocity
            x[h, i] = x[h-1, i] + v[h-1, i] * dt
        # for detecting collisions
        d = np.zeros((n, n))
        for i in range(n):
            v[h, i] = v[h-1, i]
            for j in range(i+1, n):
                d[i, j] = np.sqrt( ((x[h, i] - x[h, j])**2).sum() )
        # this approach may seem to repeat a for loop, but the structure is more extensible for adding forces
        # resolve all collisions simultaneously
        for i in range(n):
            for j in range(i+1, n):
                if d[i, j] < r[i] + r[j]:
                    if h > 1:
                        mxpre = np.sqrt(((x[h-1, i] - x[h-1, j])**2).sum())
                        if mxpre > r[i] + r[j]:
                            # first time they've encountered.
                            vp = collision2(v[h, i], v[h, j], x[h, i], x[h, j], m[i], m[j])
                            v[h, i] = vp[0]
                            v[h, j] = vp[1]
            # handle walls
            if i < k and h > 1:
                pw = pastwall(x[h, i], (wallx, lims[1]), r[i])
                pw2 = pastwall(x[h-1, i], (wallx, lims[1]), r[i])
            elif h > 1:
                pw = pastwall(x[h, i], lims, r[i])
                pw2 = pastwall(x[h-1, i], lims, r[i])
            else:
                pw = np.zeros(2)
            if pw.any():
                # hardshell collision. no forces until it escapes (in elif)
                if not isclose(pw2, pw).all():
                    # first time hitting this wall
                    invert = -(pw**2)
                    v[h, i] = v[h, i] + 2*invert*v[h, i]
            # ensure that after resolving collisions and walls that velocities point the right way
            v[h, i] = velocity_compatible(x[h, i], v[h, i], lims)


@numba.jit(nopython=True)
def hardshell_periodic(v, x, m, r, dt=1e-3, lims=(12, 12), skip=0):
    """
    Same as 'hardshell', but with periodic boundaries and the ability to only retain
    only a subset of frames with the skip keyword. E.g. skip=5 calculates 5 frames and puts the
    6th into the x and v array given.
    
    The total actual steps taken is x.shape[1] * (skip + 1)
    """
    s, n, p = x.shape
    lims = np.array(lims) # use for modular x, y
    if skip:
        padx = np.zeros((skip, n, 2))
        padv = np.zeros((skip, n, 2))
    for h in range(1, (skip + 1)*(s - 1)+1):
        sh = h % (skip+1)
        # print(h, sh)
        if skip:
            if sh > 0:
                if sh > 1:
                    oldx = padx[sh-2]
                    oldv = padv[sh-2]
                else:
                    oldx = x[h//(skip+1)]
                    oldv = v[h//(skip+1)]
            else:
                oldx = padx[-1]
                oldv = padv[-1]
        else:
            oldx = x[h-1]
            oldv = v[h-1]
        # print(oldx[:3], oldv[:3])
        newx = np.zeros((n, 2))
        newv = np.zeros((n, 2))
        for i in range(n): # for each particle
            # update position based off initial velocity
                newx[i] = (oldx[i] + oldv[i]*dt) % lims
        # for detecting collisions
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d[i, j] = periodic_distance(newx[i], newx[j], lims)
        # this approach may seem to repeat a for loop, but the structure is more extensible for adding forces
        # resolve all collisions simultaneously
        for i in range(n):
            newv = oldv
            for j in range(i+1, n):
                if d[i, j] < r[i] + r[j]:
                    if h > 1:
                        mxpre = periodic_distance(oldx[i], oldx[j], lims)
                        if mxpre > r[i] + r[j]:
                            # first time they've encountered.
                            xi = newx[i].copy()
                            xj = newx[j].copy()
                            d0 = np.abs(newx[i, 0] - newx[j, 0])
                            d1 = np.abs(newx[i, 1] - newx[j, 1])
                            if d[i, j] < np.sqrt(d0**2 + d1**2):
                                # colliding across boundary
                                if newx[i, 0] < newx[j, 0]:
                                    xj[0] = xi[0]-d0
                                else:
                                    xi[0] = xj[0]-d0
                                if newx[i, 1] < newx[j, 1]:
                                    xj[1] = xi[1]-d1
                                else:
                                    xi[1] = xj[1]-d1
                            vp = collision2(newv[i], newv[j], xi, xj, m[i], m[j])
                            newv[i] = vp[0]
                            newv[j] = vp[1]
            # ensure that after resolving collisions and walls that velocities point the right way
            newv[i] = velocity_compatible(newx[i], newv[i], lims)
        # print(newx[:3], newv[:3])
        if skip and sh > 0:
            padx[sh-1] = newx
            padv[sh-1] = newv
        else:
            x[h//(skip+1)] = newx
            v[h//(skip+1)] = newv


@numba.jit(nopython=True)
def periodic_distance(x0, x1, lims):
    d0 = np.abs(x0[..., 0] - x1[..., 0])
    d0 = np.where(d0 > lims[0]/2, lims[0] - d0, d0)
    d1 = np.abs(x0[..., 1] - x1[..., 1])
    d1 = np.where(d1 > lims[1]/2, lims[1] - d1, d1)
    return np.sqrt(d0**2 + d1**2)


@numba.jit(nopython=True)
def hardshell_harmonic_euler(v, x, m, r, c, dt=1e-3, lims=(12, 12)):
    """
    Simple harmonic force hardshell. Employs euler integration which
    gives rise to energy oscillations. You should use 'hardshell_force',
    which uses Verlet integration to dramatically reduce these oscillations.

    Parameters similar to 'hardshell_force'
    """
    s, n, p = x.shape
    for h in range(1, s):
        for i in range(n): # for each particle
            # update position based off initial velocity
            x[h, i] = x[h-1, i] + v[h-1, i] * dt
        for i in range(n):
            # reset forces
            f = np.zeros((n, 2))
            # update position based off of midpoint velocity
            for k in range(n):
                x[h, k] = x[h-1, k] + v[h-1, k]*dt
            # calculate forces
            d = np.zeros((n, n))
            for i in range(n):
                for j in range(i+1, n):
                    # check for spring pair
                    if isclose(c[i, j, 0], 0.):
                        p = -1. # sentinel value for no connection
                    else:
                        p = j
                    # get distance for possible spring / collision force
                    dx = x[h, i] - x[h, j]
                    mx = np.sqrt((dx**2).sum())
                    if j == p:
                        # apply spring force
                        ks = c[i, j, 0]
                        eq = c[i, j, 1]
                        fji = ks*(mx - eq)*dx/mx
                        f[i] -= fji
                        f[j] += fji
                    d[i, j] = mx
                # apply forces, use this new value for collisions and walls
                # need to apply forces first to conserve energy
                v[h, i] = v[h-1, i] + f[i]/m[i]*dt
            # handle collisions
            for i in range(n):
                for j in range(i+1, n):
                    if d[i,j] < r[i] + r[j]:
                        if h > 1:
                            dxpre = x[h-1, i] - x[h-1, j]
                            ddxpre = np.sqrt((dxpre**2).sum())
                            if ddxpre > r[i] + r[j]:
                                # first time they've encountered.
                                vp = collision2(v[h, i], v[h, j], x[h, i], x[h, j], m[i], m[j])
                                v[h, i] = vp[0]
                                v[h, j] = vp[1]
                # handle walls
                pw = pastwall(x[h, i], lims, r[i])
                if pw.any():
                    # hardshell collision. no forces until it escapes (in elif)
                    if h > 1:
                        pw2 = pastwall(x[h-1, i], lims, r[i])
                        if not isclose(pw2, pw).all():
                            # first time hitting this wall
                            invert = -(pw**2)
                            v[h, i] = v[h, i] + 2*invert*v[h, i]
                # ensure that after resolving collisions and walls that velocities point the right way
                v[h, i] = velocity_compatible(x[h, i], v[h, i], lims)


@numba.jit(nopython=True)
def harmonic(params, dx, mx):
    """
    params: two parameters, params[0] and params[1] that are used to calculate force
    dx: a displacement vector going from particle j to particle i
    mx: magnitude of the displacement

    Returns (force on i from j, force on j from i)
    """
    ks = params[0]  # spring constant
    eq = params[1]  # equilibrium distance
    f = ks*(mx - eq) * dx/mx  # when f is positive, there is attraction between the two particles
    return -f, f


@numba.jit(nopython=True)
def constant(params, dx, mx):
    """
    In this case params is the constant force.

    During the initialization of the forces, the diagonal entries (i,i) of "c" will
    contain the constant or environmental forces.
    """
    return params, np.zeros(2)


@numba.jit(nopython=True)
def lj(params, dx, mx):
    """
    Take two parameters, displacement vector dx from j to i, and its magnitude mx.
    Return (force on i from j, force on j from i)
    """
    ep = params[0]
    sigma = params[1]
    f = 48*ep/mx**3 * ((sigma/mx)**12 - 0.5*(sigma/mx)**6) * dx
    return f, -f


@numba.jit(nopython=True)
def hardshell_force(v, x, m, r, c, model, dt=1e-3, lims=(12, 12)):
    """
    Takes a velocity array and a position array that are 0's except for v[0] and x[0].
    Based on initial positions and velocities calculates forces according to model,
    and retains elastic colliions for overlapping particles.
    
    v: initialized velocity array of shape (steps, nparticles, 2)
    x: initialized coordinates array of shape (steps, nparticles, 2)
    m: 1d array of masses
    r: 1d array of radii
    c: (N, N, 2) array of [parameter A, parameter B] for each i, j pair of particles for the specified model
        if c[i,j] is [0.,0.] then no force is calculated for this pair of particles
        the diagonals c[i,i] are constant or 'environmental' forces on particle i.
    model: a function of the form func(params, j to i vector, distance) and returns (fij, fji)
    dt: timestep. For X's ~10 and V's ~1, dt should be ~1e-2 or less
    lims: box size, (extent in x, extent in y)
    """
    s, n, _ = x.shape  # steps, number of particles, coordinates
    vmid = np.zeros((n,2))  # for verlet integration
    f = initial_forces(x, m, r, c, model, lims=lims)
    for hs in range(1, 2*s):
        if hs%2 == 1:
            # update midpoint from full point
            for i in range(n):
                vmid[i] = v[hs//2, i] + f[i]/m[i]*0.5*dt
        else:
            # update fullpoint from midpoint
            h = hs//2
            # reset forces for new calcuations
            f = np.zeros((n, 2))
            # update position based off of midpoint velocity
            for k in range(n):
                x[h, k] = x[h-1, k] + vmid[k]*dt
            # calculate forces
            d = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    # check for interaction
                    if isclose(c[i, j], 0.).all():
                        p = -1. # no interaction, make sentinel value
                    else:
                        p = j  # there is an interaction, calculate force
                    # get distance of particles to calculate force or check for collision
                    dx = x[h, i] - x[h, j]
                    mx = np.sqrt((dx**2).sum())
                    d[i, j] = mx
                    if j == p:
                        # update forces, dx is vector from j to i, with magnitude mx
                        fij, fji = model(c[i, j], dx, mx)
                        f[i] += fij # force on i from j
                        f[j] += fji # forve on j from i
                # apply forces, use this new value for collisions and walls
                # need to apply forces before collisions to conserve energy
                v[h, i] = vmid[i] + f[i]/m[i]*0.5*dt
            # handle collisions
            for i in range(n):
                for j in range(i+1, n):
                    if d[i,j] < r[i] + r[j]:
                        if h > 1:
                            dxpre = x[h-1, i] - x[h-1, j]
                            ddxpre = np.sqrt((dxpre**2).sum())
                            if ddxpre > r[i] + r[j]:
                                # first time they've encountered.
                                vp = collision2(v[h, i], v[h, j], x[h, i], x[h, j], m[i], m[j])
                                v[h, i] = vp[0]
                                v[h, j] = vp[1]
                # handle walls
                pw = pastwall(x[h, i], lims, r[i])
                if pw.any():
                    # hardshell collision. no forces until it escapes (in elif)
                    if h > 1:
                        pw2 = pastwall(x[h-1, i], lims, r[i])
                        if not isclose(pw2, pw).all():
                            # first time hitting this wall
                            invert = -(pw**2)
                            v[h, i] = v[h, i] + 2*invert*v[h, i]
                # ensure that after resolving collisions and walls that velocities point the right way
                v[h, i] = velocity_compatible(x[h, i], v[h, i], lims)


@numba.jit(nopython=True)
def initial_forces(x, m, r, c, model, lims=(12, 12)):
    s, n, _ = x.shape  # steps, number of particles, coordinates
    # initialize 0's
    f = np.zeros((n, 2))
    # update fullpoint from midpoint
    h = 0
    # calculate forces
    for i in range(n):
        for j in range(i, n):
            # check for spring pair
            if isclose(c[i, j], 0.).all():
                p = -1. # sentinel value for no connection
            else:
                p = j
            # get distance for possible spring / collision force
            dx = x[h, i] - x[h, j]
            mx = np.sqrt((dx**2).sum())
            if mx < r[i] + r[j]:
                continue
            elif j == p:
                # apply force
                fij, fji = model(c[i, j], dx, mx)
                f[i] += fij
                f[j] += fji
        pw = pastwall(x[h, i], lims, r[i])
        if pw.any():
            # hardshell collision. no forces until it escapes (in elif)
            f[i] = 0.
    return f


