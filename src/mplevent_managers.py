import matplotlib.pyplot as plt

class PickManager:
    """Base class for managing pick events on a figure. Note that each
    registered picker function is called after a pick event. Make sure you
    register functions that return NoneType if type is wrong.

    Parameters
    ----------
    fig : matplotlib.Figure

    Example
    ------
    >>> from numpy.random import rand
    >>> fig, ax = plt.subplots()
    >>> pm = PickManager(fig)
    >>> x = rand(100)
    >>> v, b, p = ax.hist(x, picker=1)
    >>> def select_bin(event, ax=ax):
    >>>     print('selecting bin: ', event.artist)
    >>>     rect = event.artist
    >>>     rect.set_facecolor('red')
    >>>     fig.canvas.draw()
    >>> def select_alpha(event, ax=ax):
    >>>     print('changing alpha: ', event.artist)
    >>>     rect = event.artist
    >>>     rect.set_alpha(0.5)
    >>>     fig.canvas.draw()
    >>> pm.add_pick_callback(select_bin)
    >>> pm.add_pick_callback(select_alpha)
    >>> plt.show()
    """
    def __init__(self, fig):
        self.fig = fig
        self._pickers = {}

    def add_pick_callback(self, picker):
        assert callable(picker)
        assert picker not in self._pickers.values()
        pickcid = self.fig.canvas.mpl_connect('pick_event', picker)
        self._pickers[pickcid] = picker

    def connect_picks(self):
        # cid's are integers. disconnected are strs.
        for pickcid in self._pickers:
            if isinstance(pickcid, str):
                newcid = self.fig.canvas.mpl_connect(self._pickers[pickcid])
                self._pickers[newcid] = self._pickers[pickcid]
                del self._pickers[pickcid]

    def disconnect_picks(self):
        # cid's are integers. disconnected are strs.
        for pickcid in self._pickers:
            if not isinstance(pickcid, str):
                self.fig.canvas.mpl_disconnect(pickcid)
                self._pickers[str(pickcid)] = self._pickers[pickcid]
                del self._pickers[pickcid]

    def clear_picks(self):
        self._pickers = {}


class KeyMapManager:
    """Base class for managing keys on matplot figures

    Parameters
    ----------
    fig : matplotlib.Figure

    Example
    -------
    Note: mostly used for subclassing. But the following works.
    >>> from numpy.random import rand
    >>> fig, ax = plt.subplots()
    >>> km = KeyMapManager(fig)
    >>> x = rand(100)
    >>> v, b, p = ax.hist(x)
    >>> def resamp(event, ax=ax):
    >>>     print('resampling!')
    >>>     ax.cla()
    >>>     v, b, p = ax.hist(rand(100))
    >>>     fig.canvas.draw()
    >>> km.add_key_callback('r', 'Resample data', resamp)
    >>> plt.show()
    >>> km  # see repr
    """
    def __init__(self, fig):
        self.fig = fig
        # Deactivate the default keymap
        keypressid = fig.canvas.manager.key_press_handler_id
        fig.canvas.mpl_disconnect(keypressid)
        # Activte keymap
        self._keymap = {}
        self.connect_keymap()
        self._lastkey = None

    def connect_keymap(self):
        self._keycid = self.fig.canvas.mpl_connect('key_press_event', self.keypress)

    def keypress(self, event):
        description, callback = self._keymap.get(event.key, (None, None))
        if callback:
            self._lastkey = event.key
            callback(event)

    def disconnect_keymap(self):
        if self._keycid is not None:
            self.fig.canvas.mpl_disconnect(self._keycid)
            self._keycid = None

    def add_key_callback(self, key, description, callback):
        assert callable(callback)
        assert key not in self._keymap
        self._keymap[key] = (description, callback)

    def clear_keymap(self):
        self._keymap = {}

    def help(self):
        key = lambda k, d: "Key: {}  Description: {}\n".format(k, d)
        msg = ''.join([key(k, d[0]) for k, d in self._keymap.items()])
        return repr(self.fig) + '\n' + msg


class MouseManager:
    
    def __init__(self, fig):
        self.fig = fig
        self._mousefuncs = {}

    def add_mouse_callback(self, button, callback):
        """
        Parameters
        ----------
        button : 'press' or 'scroll'
            'up' and 'down' are scroll buttons
        """
        assert callable(callback)
        assert callback not in self._mousefuncs.values()
        if button == 'press':
            cid = self.fig.canvas.mpl_connect('button_press_event', callback)
        elif button == 'scroll':
            cid = self.fig.canvas.mpl_connect('scroll_event', callback)
        else:
            raise ValueError("buttum not valid")
        self._mousefuncs[cid] = callback

    # connect and disconnet allow for toggling. 
    def connect_mouse(self):
        # cid's are integers. disconnected are strs.
        for cid in self._callbacks:
            if isinstance(cid, str):
                newcid = self.fig.canvas.mpl_connect(self._callbacks[cid])
                self._callbacks[newcid] = self._callbacks[cid]
                del self._callbacks[cid]

    def disconnect_mouse(self):
        # cid's are integers. disconnected are strs.
        for cid in self._callbacks:
            if not isinstance(cid, str):
                self.fig.canvas.mpl_disconnect(cid)
                self._callbacks[str(cid)] = self._callbacks[cid]
                del self._callbacks[cid]

    def clear_mouse(self):
        self._callbacks = {}
