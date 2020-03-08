import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import deque
from matplotlib.widgets import Button
from pandas import DataFrame
from itertools import combinations
from mplevent_managers import *
from matplotlib.widgets import RectangleSelector


class FrameView(KeyMapManager, MouseManager):
    def __init__(self, ims,  extent='auto', patch_maker=None, in_range='full', origin='upper',
                 time_unit = '', time_loc='upper left', frames=None, times=None, framemaster=None,
                 pixel_cmap='gray', font_dict={'size': 12}, rectangle=False):
        """
        Simple event driven frameview bound to right and left key presses.
        Parameters
        ----------
        ims : ndimage (frame, rows, columns)
        extent : ['auto', (left, right, bottom, top), pd.DataFrame indexed by frames]
            The coordinates of the image
        patch_maker : callable
            Accepts a frame argument and returns a list of patches to add
        in_range : ['full', 'auto']
        interval : (int, str)
            Number of units per frame increment. 'E.g.' (2, 'sec')
        time_unit : 'sec', 'min', ...
        time_loc : ['lower left', ..., 'upper right']
            Coordinates of time location. Set to None to have no time stamp.
        pixel_cmap : str
            colormap for pixel data.
        rectangle : bool
            Whether to hook in a rectangle drawing widget for potential cropping functions.
        Example
        -------
        >>> import numpy as np
        >>> ims = np.random.randint(0, 255, (10, 256, 256), dtype=np.uint8)
        >>> fig, ax = plt.subplots()
        >>> fv = FrameView(ims)
        >>> plt.show()
        """
        self.ims = ims
        if frames is None:
            self.frames = deque(range(len(ims)))
        else:
            self.frames = deque(frames)
        self._frames = np.array(self.frames)
        assert len(self.frames) == len(ims)
        self.ims = {k: i for k, i in zip(self.frames, ims)}
        if framemaster is None:
            self.framemaster = None
            self.stepsize = int(len(self._frames) // 20)
        else:
            self.framemaster = deque(framemaster)
            self.stepsize = len(framemaster) // 20
        self.frame = self.frames[0]
        if times is None:
            self.times = deque(list(self.frames))
        else:
            self.times = deque(times)
        self.time_unit = time_unit
        self.fd = font_dict
        # We want a patch generator since it is low on memory (a tad higher in processing)
        # and we want fresh patches to avoid errors from using patches on different axes
        # This is relevant since self.mp4 requires "Agg" backend associated canvas, which we
        # don't want to interfere with pyplot's global state.
        self.patch_maker = patch_maker
        self.ax = plt.gca()
        self.fig = plt.gcf()
        if hasattr(self.ax, 'get_subplotspec'):
            # If using a gridspec, later use of fig.subplots_adjust is ignored
            ss = self.ax.get_subplotspec()
            gs = ss.get_gridspec()
            gs.update(left=0, right=1, bottom=0, top=1)
        else:
            self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax.axis('off')

        self.start = 0
        self.stop = len(ims)-1
        if isinstance(extent, str) and extent == 'auto':
            try:
                shape = ims.shape
            except AttributeError:  # Maybe a PIMS frame object
                shape = (len(ims), *ims.frame_shape)
            self.extent = (0, shape[2], shape[1], 0)
            self.dfextent = None
        elif isinstance(extent, pd.DataFrame):
            self.dfextent = extent
            self.extent = self.dfextent.iloc[0]
        elif isinstance(extent, (tuple, list)):
            self.dfextent = None
            self.extent = extent
        self.pmap = pixel_cmap
        # Display ranges
        self.in_range = in_range
        if in_range == 'full':
            mn, mx = (int(ims.min()), int(ims.max()))
            self.vmin, self.vmax = mn, mx
        elif in_range == 'auto':
            self.vmin, self.vmax = None, None
        self.imax = self.ax.imshow(ims[0], self.pmap, vmin=self.vmin, vmax=self.vmax,
                                   interpolation='none', extent=self.extent, origin=origin)
        self.ax.axis('off')
        self.fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        self.time_loc = time_loc
        self.time_stamp = None
        self.set_time()
        self.add_patches()
        KeyMapManager.__init__(self, self.fig)
        MouseManager.__init__(self, self.fig)
        # Add Key Functions
        self.add_key_callback('right', 'Increment frame', self.handle_right)
        self.add_key_callback('left', 'Decrement frame', self.handle_left)
        # Add Scroll Functions
        self.add_mouse_callback('scroll', self.handle_scroll)
        # Add Rectangl e Selector
        if rectangle:
            self.rs = RectangleSelector(self.ax, self.line_select_callback,
                                        drawtype='box', useblit=True,
                                        button=[1, 3],  # don't use middle button
                                        minspanx=10, minspany=10,
                                        spancoords='pixels',
                                        interactive=True)
        else:
            self.rs = None
        # Other
        self.frame_callbacks = []
        self.updaters = []
    def line_select_callback(self, eclick, erelease):
        'eclick and erelease are the press and release events'
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        print(" The buttons you used were: %s %s" % (eclick.button, erelease.button))
        if len(self.ax.lines) > 2: [l.remove() for l in self.ax.lines[:-2]]
    def add_frame_callback(self, callback):
        """Functions of the from foo(image, ax) -> ax"""
        if callback not in self.frame_callbacks:
            self.frame_callbacks.append(callback)
    def rotate(self, step):
        if self.framemaster:
            self.framemaster.rotate(step)
            fr = self.framemaster[0]
            s = int(step/abs(step))
            if fr in self.frames:
                it = 0
                while fr != self.frames[0]:
                    self.frames.rotate(s)
                    self.times.rotate(s)
                    it += 1
                    if it > len(self.frames):
                        raise ValueError("fr somehow in self.frames")
                self.frame = self.frames[0]
            else:
                # display most recent frame
                df = fr - self._frames
                if all(df < 0):
                    mindf = (-df).min()
                    minf = self._frames[df == mindf]
                else:
                    mindf = df[df > 0].min()
                    minf = self._frames[df == mindf]
                it = 0
                while minf != self.frames[0]:
                    self.frames.rotate(1)
                    self.times.rotate(1)
                    it += 1
                    if it > len(self.frames):
                        raise ValueError("fr somehow in self.frames")
                self.frame = self.frames[0]
        else:
            self.frames.rotate(step)
            self.times.rotate(step)
            self.frame = self.frames[0]
    def handle_scroll(self, event):
        step = int(event.step * self.stepsize)
        self.rotate(step)
        self.update(event)
    def handle_right(self, event):
        self.rotate(-1)
        self.update(event)
    def handle_left(self, event):
        self.rotate(1)
        self.update(event)
    def update(self, event):
        # Ordering here is important. We want to draw the pixels first AND THEN the patches
        # otherwise the pixels will cover that patch.
        # Patches first: points, etc.
        for p in reversed(self.ax.patches): p.remove()
        for a in reversed(self.ax.artists): a.remove()
        # Artists: timestamps, labels, etc.
        if self.in_range == 'auto':
            self.show_image()
        else:
            self.set_image() # recalculate display values
        self.set_extent()  # If the extent changes each frame
        self.set_time()
        self.add_patches()
        if self.frame_callbacks:
            for cb in self.frame_callbacks:
                cb(self.ims[self.frame - self.start], self.ax)
        if self.updaters:
            for ud in self.updaters:
                ud()
        self.ax.figure.canvas.draw()
        if self.rs: self.rs.update()
    def show_image(self):
        image = self.ims[self.frame]
        self.imax = self.ax.imshow(image, self.pmap, vmin=self.vmin, vmax=self.vmax,
                                   interpolation='none', extent=self.extent)
    def set_image(self):
        image = self.ims[self.frame]
        self.imax.set_data(image)
    def set_extent(self):
        if self.dfextent is not None:
            new_extent = self.dfextent.loc[self.frame].values
            self.extent = new_extent
            self.imax.set_extent(new_extent)
        else:
            pass
    def add_patches(self):
        if self.patch_maker is not None:
            ps = self.patch_maker(self.frame)
            if ps:
                for p in ps:
                    self.ax.add_patch(p)
    def set_time(self):
        if self.time_loc is not None:
            if self.times is None:
                label = f'frame {self.frame}'
            else:
                label = f'{self.times[0]:.02f} {self.time_unit}\nframe: {self.frame}'
            self.time_stamp = self.ax.set_title(label)


class FrameViewYN(FrameView):
    def __init__(self, *args, **kwargs):
        """use figsize with vertical dimension 1.2 that of normal"""
        self.msg = kwargs.pop('msg', "Yes or No?")
        self.answer = None
        super().__init__(*args, **kwargs)
        # make room for buttons
        w, h = self.fig.get_size_inches()
        self.fig.set_size_inches(w, w*1.2)
        if hasattr(self.ax, 'get_subplotspec'):  # If using a gridspec, fig.subplots_adjust is ignored
            ss = self.ax.get_subplotspec()
            gs = ss.get_gridspec()
            gs.update(bottom=0.2)
        else:
            self.fig.subplots_adjust(bottom=0.2)
        axno = self.fig.add_axes([0.17, 0.05, 0.2, 0.1])
        axys = self.fig.add_axes([0.66, 0.05, 0.2, 0.1])
        self.bno = Button(axno, 'No')
        self.bno.on_clicked(self.no)
        self.bys = Button(axys, 'Yes')
        self.bys.on_clicked(self.yes)
        self.fig.canvas.set_window_title(self.msg)
    def no(self, event):
        self.answer = 'no'
        plt.close(self.fig)
    def yes(self, event):
        self.answer = 'yes'
        plt.close(self.fig)

if __name__ == '__main__':
    import numpy as np
    ims = np.random.randint(0, 255, (10, 256, 256), dtype=np.uint8)
    fig, ax = plt.subplots()
    fv = FrameView(ims)
    plt.show()
