import math
import random
import matplotlib.pyplot as plt
import numpy as np
import numba
import matplotlib.animation as animation
from matplotlib import colors


def simulate_cme(Q, V, time, s0):
    """
    Algorithm from Gillespie 2007
    
    inputs
    ------
    Q: Q(current_state) -> vector of transition rates
    V: stoichiometry matrix, rows reaction, column number changes
    time: stop simulation after 'time' has passed
    s0: initial state vector
    
    returns
    -------
    C: evolution of s0 over time
    T: array of times with transitions occured
    """
    clock = 0
    current_state = s0
    transitions = [current_state]
    times = [0]
    i = 0
    while clock < time:
        Qs = Q(current_state)
        a0 = sum(Qs)
        if a0:
            tao = - 1/a0 * math.log(random.random())
            m = random.random() * a0
        else:
            # terminate
            clock += time - clock
            times.append(clock)
            transitions.append(current_state)
            print("NO transition rates, terminating")
            continue
        j = (Qs.cumsum() - m > 0)
        if any(j):
            j = j.argmax()
        else:
            j = 0
        new_state = current_state + V[j]
        transitions.append(new_state)
        current_state = new_state
        clock += tao
        times.append(clock)
        i += 1
        if i > 1e6:
            print("Iteration limit reached")
            break
    return np.vstack(transitions), np.array(times)


# def generate_cme(Q, V, time, s0):
#     """
#     Algorithm from Gillespie 2007
    
#     inputs
#     ------
#     Q: Q(current_state) -> vector of transition rates
#     V: stoichiometry matrix, rows reaction, column number changes
#     time: stop simulation after 'time' has passed
#     s0: initial state vector
    
#     returns
#     -------
#     C: evolution of s0 over time
#     T: array of times with transitions occured
#     """
#     clock = 0
#     current_state = s0
#     yield current_state, clock
#     i = 0
#     while clock < time:
#         Qs = Q(current_state)
#         a0 = sum(Qs)
#         if a0:
#             tao = - 1/a0 * math.log(random.random())
#             m = random.random() * a0
#         else:
#             # terminate
#             clock += time - clock
#             times.append(clock)
#             transitions.append(current_state)
#             print("NO transition rates, terminating")
#             continue
#         j = (Qs.cumsum() - m > 0)
#         if any(j):
#             j = j.argmax()
#         else:
#             j = 0
#         current_state = current_state + V[j]
#         clock += tao
#         i += 1
#         if i > 1e6:
#             print("Iteration limit reached")
#             break
#         yield current_state, clock


@numba.jit(nopython=True)
def runtime_cme(Q, V, time, s0):
    """
    Algorithm from Gillespie 2007
    
    inputs
    ------
    Q: Q(current_state) -> vector of transition rates
    V: stoichiometry matrix, rows reaction, column number changes
    time: stop simulation after 'time' has passed
    s0: initial state vector
    
    returns
    -------
    C: evolution of s0 over time
    T: array of times with transitions occured
    """
    clock = 0
    current_state = s0
    i = 0
    while clock < time:
        Qs = Q(current_state)
        a0 = Qs.sum()
        if a0:
            tao = - 1/a0 * math.log(random.random())
            m = random.random() * a0
        else:
            # terminate
            clock += time - clock
            print("NO transition rates, terminating")
            continue
        j = (Qs.cumsum() - m > 0)
        if np.any(j):
            ix = j.argmax()
        else:
            ix = 0
        current_state = current_state + V[ix]
        clock += tao
        i += 1
        if i > 1e6:
            print("Iteration limit reached")
            break
    return current_state, clock


def simulate_cmc(Q, time, s0=0):
    clock = 0  # Keep track of the clock
    current_state = s0  # First state
    #  time_spent = {s:0 for s in state_space}
    transitions = [current_state]  # start in state 0
    times = [0]
    while clock < time:
        # Sample the transitions
        Qs = Q[current_state]
        sojourn_times = np.random.exponential(scale=1/np.abs(Qs))
        sojourn_times = np.where(Qs < 0, np.inf, sojourn_times)

        # Identify the next state
        # next_state = min([state_space, key=lambda x: sojourn_times[x])
        next_state = sojourn_times.argmin()
        sojourn = sojourn_times[next_state]
        clock += sojourn
        times.append(clock)
        transitions.append(next_state)
        # if clock > warm_up:  # Keep track if past warm up time
        #     time_spent[current_state] += sojourn
        current_state = next_state  # Transition

#     pi = [time_spent[state] / sum(time_spent.values()) for state in state_space]  # Calculate probabilities
    return transitions, times


def states_generator(Q, time=10, nx=10, sr=5) -> (float, np.ndarray):
    N = nx ** 2
    states = np.zeros((nx, nx), dtype=np.uint8)
    rvs = np.ones((nx, nx)) * np.inf
    mk = Q.min() # find state with fastest escape rate (-'s on diagonal)
    step = abs(1/sr * 1 / mk) # nyquist sample fastest averate rate
    t = 0.
    yield t, states
    while t < time:
        Qs = Q[states]
        rvs = np.random.exponential(scale=1/np.abs(Qs))
        rvs = np.where(Qs < 0, np.inf, rvs)
        sojourn_times = rvs.min(axis=-1)
        states = np.where(sojourn_times < step, rvs.argmin(axis=-1), states)
        t += step
        yield t, states


cmap = colors.ListedColormap(['goldenrod', 'royalblue'])

def liveCMC(Q, time, nx, sr=5, ticks=False):
    bounds = [0,1]
    # norm = colors.BoundaryNorm(bounds, cmap.N)  # use with ax.imshow(..., norm=norm) remaps to [0,1]
    
    fig, ax = plt.subplots(figsize=(6,6))
    extent = (0, nx, nx, 0)

    sgen = states_generator(Q, time, nx, sr)
    t, s = next(sgen)
    imax = ax.imshow(s, cmap=cmap, extent=extent, origin='upper')
    ax.set_title(str(t))
    
    # draw gridlines
    if ticks:
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(0, extent[1], 1))
        ax.set_yticks(np.arange(0, extent[2], 1))
    else:
        ax.axis('off')
    
    num = [s.sum()]
    
    def advance(g):
        t, s = g[0], g[1]
        imax.set_data(s)
        num.append(s.sum())
        imax.autoscale()  # if first frame only has one color, subsequent frames won't update color unless this is here
        ax.set_title(f'{t:.02f} sec')
    
    fa = animation.FuncAnimation(fig, func=advance, frames=sgen)
    return fa, np.array(num)


def CMC(Q, time, nx, sr=5, ticks=False):
    sgen = states_generator(Q, time, nx, sr)
    num = []
    ts = []
    for t, s in sgen:
        num.append(s.sum())
        ts.append(t)
    return np.array(ts), np.array(num)

@numba.jit(nopython=True)
def dot(m, v):
    y = np.zeros(len(v), dtype=np.float64)
    for i in range(m.shape[0]):
        y[i] = (m[i] * v).sum()
    return y
