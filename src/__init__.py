"""
Author: Dario ML, bdevans
Program: src/__init__.py
Description: main file for python-ae
"""

from __future__ import print_function
from PIL import Image
import math
import numpy as np
import scipy.io
import os
import time
import matplotlib.pyplot as plt
from matplotlib import cm

# TODO: Generalise to process off-events
# TODO: Add function to combine on and off-events


class aefile(object):
    def __init__(self, filename, max_events=1e6):
        self.filename = filename
        self.max_events = max_events
        self.header = []
        self.data, self.timestamp = self.read()

    # alias for read
    def load(self):
        return self.read()

    def read(self):
        with open(self.filename, 'r') as f:
            line = f.readline()
            while line[0] == '#':
                self.header.append(line)
                if line[0:9] == '#!AER-DAT':
                    aer_version = line[9]
                current = f.tell()
                line = f.readline()

            if aer_version != '2':
                raise Exception('Invalid AER version. Expected 2, got %s' % aer_version)

            f.seek(0, 2)
            numEvents = math.floor((f.tell() - current) / 8)

            if numEvents > self.max_events:
                print('There are %i events, but max_events is set to %i. Will only use %i events.' % (numEvents, self.max_events, self.max_events))
                numEvents = self.max_events

            f.seek(current)

            timestamps = np.zeros(numEvents)
            data = np.zeros(numEvents)

            for i in range(int(numEvents)):
                data[i] = int(f.read(4).encode('hex'), 16)
                timestamps[i] = int(f.read(4).encode('hex'), 16)

            return data, timestamps

    def save(self, data=None, filename=None, ext='aedat'):
        if filename is None:
            filename = self.filename
        if data is None:
            data = aedata(self)
        if ext is 'aedat':
            # unpack our 'data'
            ts = data.ts
            data = data.pack()

            with open(filename, 'w') as f:
                for item in self.header:
                    f.write(item)
                # print('\n\n')
                # f.write('\n\n')  # Was this meant to write to the file?
                num_items = len(data)
                for i in range(num_items):
                    f.write(hex(int(data[i]))[2:].zfill(8).decode('hex'))
                    f.write(hex(int(ts[i]))[2:].zfill(8).decode('hex'))

    def unpack(self):
        noData = len(self.data)
        x = np.zeros(noData)
        y = np.zeros(noData)
        t = np.zeros(noData)

        for i in range(noData):
            d = int(self.data[i])
            t[i] = d & 0x1
            x[i] = 128-((d >> 0x1) & 0x7F)
            y[i] = (d >> 0x8) & 0x7F
        return x, y, t


class aedata(object):
    def __init__(self, ae_file=None):
        self.dimensions = (128, 128)
        if isinstance(ae_file, aefile):
            self.x, self.y, self.t = ae_file.unpack()
            self.ts = ae_file.timestamp
        elif isinstance(ae_file, aedata):
            self.x, self.y, self.t = aedata.x, aedata.y, aedata.t
            self.ts = ae_file.ts
        else:
            self.x, self.y = np.array([]), np.array([])
            self.t, self.ts = np.array([]), np.array([])

    def __getitem__(self, item):
        rtn = aedata()
        rtn.x = self.x[item]
        rtn.y = self.y[item]
        rtn.t = self.t[item]
        rtn.ts = self.ts[item]
        return rtn

    def __setitem__(self, key, value):
        self.x[key] = value.x
        self.y[key] = value.y
        self.t[key] = value.t
        self.ts[key] = value.ts

    def __delitem__(self, key):
        self.x = np.delete(self.x, key)
        self.y = np.delete(self.y, key)
        self.t = np.delete(self.t, key)
        self.ts = np.delete(self.ts, key)

    def save_to_mat(self, filename):
        scipy.io.savemat(filename, {'X': self.x, 'Y': self.y,
                                    't': self.t, 'ts': self.ts})

    def pack(self):
        noData = len(self.x)
        packed = np.zeros(noData)
        for i in range(noData):
            packed[i] = (int(self.t[i]) & 0x1)
            packed[i] += (int(128-self.x[i]) & 0x7F) << 0x1
            packed[i] += (int(self.y[i]) & 0x7F) << 0x8

        return packed

    # TODO
    # performance here can be improved by allowing indexing in the AE data.
    # For now, I expect this not to be done often
    def make_sparse(self, ratio):
        indexes = np.random.randint(0, len(self.x),
                                    math.floor(len(self.x) / ratio))
        indexes.sort()

        rtn = aedata()
        rtn.x = self.x[indexes]
        rtn.y = self.y[indexes]
        rtn.t = self.t[indexes]
        rtn.ts = self.ts[indexes]

        return rtn

    def __repr__(self):
        return "%i total [x,y,t,ts]: [%s, %s, %s, %s]" % (len(self.x), self.x, self.y, self.t, self.ts)

    def __len__(self):
        return len(self.x)

    def interactive_animation(self, step=5000, limits=(0, 128), pause=0):
        plt.ion()
        fig = plt.figure(figsize=(6, 6))
        plt.show()
        ax = fig.add_subplot(111)

        start = 0
        end = step - 1
        while(start < len(self.x)):
            ax.clear()
            ax.scatter(self.x[start:end], self.y[start:end],
                       s=20, c=self.t[start:end], marker='o', cmap=cm.jet)
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            start += step
            end += step
            plt.draw()
            time.sleep(pause)

    def downsample(self, new_dimensions=(16, 16)):
        # TODO
        # Make this cleaner
        assert self.dimensions[0] % new_dimensions[0] is 0
        assert self.dimensions[1] % new_dimensions[1] is 0

        rtn = aedata()

        rtn.ts = self.ts
        rtn.t = self.t
        rtn.x = np.floor(self.x / (self.dimensions[0] / new_dimensions[0]))
        rtn.y = np.floor(self.y / (self.dimensions[1] / new_dimensions[1]))

        return rtn

    def to_matrix(self, dim=(128, 128)):
        return make_matrix(self.x, self.y, self.t, dim=dim)


def make_matrix(x, y, t, dim=(128, 128)):
    image = np.zeros(dim)
    events = np.zeros(dim)

    for i in range(len(x)):
        image[y[i]-1, x[i]-1] -= t[i]-0.5
        events[y[i]-1, x[i]-1] += 1

    # http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
    np.seterr(divide='ignore', invalid='ignore')

    result = 0.5 + (image / events)
    result[events == 0] = 0.5
    return result


def create_pngs(data, prepend, path="", step=3000, dim=(128, 128)):
    if not os.path.exists(path):
        os.makedirs(path)

    idx = 0
    start = 0
    end = step - 1
    while(start < len(data.x)):
        image = make_matrix(data.x[start:end], data.y[start:end],
                            data.t[start:end], dim=dim)
        img_arr = (image*255).astype('uint8')
        im = Image.fromarray(img_arr)
        im.save(path + os.path.sep + prepend + ("%05d" % idx) + ".png")
        idx += 1

        start += step
        end += step


def concatenate(a_tuple):
    rtn = aedata()
    rtn.x = np.concatenate(tuple([a_tuple[i].x for i in range(len(a_tuple))]))
    rtn.y = np.concatenate(tuple([a_tuple[i].y for i in range(len(a_tuple))]))
    rtn.t = np.concatenate(tuple([a_tuple[i].t for i in range(len(a_tuple))]))
    rtn.ts = np.concatenate(tuple([a_tuple[i].ts for i in range(len(a_tuple))]))
    return rtn

    # np.concatenate(a_tuple)
