import matplotlib
import matplotlib.pyplot as plt

def flatten(iterable):
    if not hasattr(iterable, '__iter__'):
        print("Arg doens't have attr '__iter__', returning")
        return iterable
    typ = type(iterable)  # return tuple or list type
    i = 0
    lst = list(iterable)
    while i < len(lst):
        while getattr(lst[i], '__iter__', None):
            # pop empty items
            try:
                lst[i][0]
                lst[i:i+1] = lst[i]
            except IndexError:
                lst.pop(i)
                i -= 1
                break
        i += 1
    # could try returning type with typ(lst), but many
    # non standard containers (np.ndarray, etc.) fail.
    return lst


def add_alpha(c, alpha):
    if len(c) == 3:
        c.append(alpha)
    elif len(c) == 4:
        c[3] = alpha
    else:
        print("color not of length 3 or 4")
    return c


def hard_edge(patches, alpha, edge='same', lw=plt.rcParams['lines.linewidth']):
    """
    patches: 3rd output of matplotlib's hist(). _, _, patches = hist()
    alpha: the desired transparancy
    edge: 'same' means edge will be the same color. Can also do edge='black'
    lw: the linewidth of the edge
    """
    ps = flatten(patches)
    for p in ps:
        fc = list(p.get_facecolor())
        if edge  == 'same':
            ec = fc[:]
        else:
            ec = [0, 0, 0, 1]
        fca = add_alpha(fc, alpha)
        eca = add_alpha(ec, 1)
        p.set_facecolor(fca)
        p.set_edgecolor(eca)
        p.set_linewidth(lw)
