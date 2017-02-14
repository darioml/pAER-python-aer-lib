"""
Author: Dario ML, bdevans
Program: src/__init__.py
Description: main file for python-ae
"""

from __future__ import print_function
import copy
import math
import os
import time
from PIL import Image
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import cm


class AEFile(object):
    """Class representing AER data file"""
    def __init__(self, filename, max_events=1e6):
        self.filename = filename
        self.max_events = max_events
        self.header = []
        self.data, self.timestamp = self.read()

    def load(self):
        """Alias for read"""
        return self.read()

    def read(self):
        """Read data from file, ignoring comments"""
        with open(self.filename, 'r') as aef:
            line = aef.readline()
            while line[0] == '#':
                self.header.append(line)
                if line[0:9] == '#!AER-DAT':
                    aer_version = line[9]
                current = aef.tell()
                line = aef.readline()

            if aer_version != '2':
                raise Exception('Invalid AER version. '
                                'Expected 2, got {}'.format(aer_version))

            aef.seek(0, 2)
            num_events = math.floor((aef.tell() - current) / 8)

            if num_events > self.max_events:
                print('There are {} events, but max_events is set to {}. '
                      'Will only use {} events.'.format(num_events,
                                                        self.max_events,
                                                        self.max_events))
                num_events = self.max_events

            aef.seek(current)
            j = 0
            timestamps = np.zeros(num_events)
            data = np.zeros(num_events)

            # print(num_events)
            for i in range(int(num_events)):
                aef.seek(current+8*i)
                # data[i] = int(aef.read(4).encode('hex'), 16)
                # timestamps[i] = int(aef.read(4).encode('hex'), 16)

                cur_data = int(aef.read(4).encode('hex'), 16)
                cur_timestamp = int(aef.read(4).encode('hex'), 16)
                if j > 0:
                    time_diff = cur_timestamp - timestamps[j-1]
                    if 0 <= time_diff <= 1e8:
                        data[j] = cur_data
                        timestamps[j] = cur_timestamp
                        j += 1

                elif j == 0:
                    data[j] = cur_data
                    timestamps[j] = cur_timestamp
                    j += 1

            return data, timestamps

    def save(self, data=None, filename=None, ext='aedat'):
        """Saves given data to file name"""
        if filename is None:
            filename = self.filename
        if data is None:
            data = AEData(self)
        if ext is 'aedat':
            # unpack our 'data'
            timestamp = data.timestamp
            data = data.pack()

            with open(filename, 'w') as aef:
                for item in self.header:
                    aef.write(item)
                # print('\n\n')
                # aef.write('\n\n')  # Was this meant to write to the file?
                current = aef.tell()
                num_items = len(data)
                for i in range(num_items):
                    aef.seek(current+8*i)
                    aef.write(hex(int(data[i]))[2:].zfill(8).decode('hex'))
                    aef.write(hex(int(timestamp[i]))[2:].zfill(8).decode('hex'))

    def unpack(self):
        """Unpack x, y axes and time data from loaded data"""
        no_data = len(self.data)
        x_data = np.zeros(no_data)
        y_data = np.zeros(no_data)
        t_data = np.zeros(no_data)

        for idx, data in enumerate(no_data):
            t_data[idx] = data & 0x1
            x_data[idx] = 128-((data >> 0x1) & 0x7F)
            y_data[idx] = (data >> 0x8) & 0x7F
        return x_data, y_data, t_data


class AEData(object):
    """Class representation of AER data"""
    def __init__(self, ae_file=None):
        self.dimensions = (128, 128)
        if isinstance(ae_file, AEFile):
            self.x_data, self.y_data, self.t_data = ae_file.unpack()
            self.timestamp = ae_file.timestamp
        elif isinstance(ae_file, AEData):
            self.x_data, self.y_data, self.t_data = (ae_file.x_data,
                                                     ae_file.y_data,
                                                     ae_file.t_data)
            self.timestamp = ae_file.timestamp
        else:
            self.x_data, self.y_data = np.array([]), np.array([])
            self.t_data, self.timestamp = np.array([]), np.array([])

    def __getitem__(self, item):
        rtn = AEData()
        rtn.x_data = self.x_data[item]
        rtn.y_data = self.y_data[item]
        rtn.t_data = self.t_data[item]
        rtn.timestamp = self.timestamp[item]
        return rtn

    def __setitem__(self, key, value):
        self.x_data[key] = value.x_data
        self.y_data[key] = value.y_data
        self.t_data[key] = value.t_data
        self.timestamp[key] = value.timestamp

    def __delitem__(self, key):
        self.x_data = np.delete(self.x_data, key)
        self.y_data = np.delete(self.y_data, key)
        self.t_data = np.delete(self.t_data, key)
        self.timestamp = np.delete(self.timestamp, key) 

    def save_to_mat(self, filename):
        """Save object data to .mat file using scipy"""
        scipy.io.savemat(filename, {'X': self.x_data, 'Y': self.y_data,
                                    't': self.timestamp, 'ts': self.timestamp})

    def pack(self):
        """Packs data into binary format"""
        no_data = len(self.x_data)
        packed = np.zeros(no_data)
        for i in range(no_data):
            packed[i] = (int(self.t_data[i]) & 0x1)
            packed[i] += (int(128-self.x_data[i]) & 0x7F) << 0x1
            packed[i] += (int(self.y_data[i]) & 0x7F) << 0x8

        return packed

    # TODO
    # performance here can be improved by allowing indexing in the AE data.
    # For now, I expect this not to be done often
    def make_sparse(self, ratio):
        """Random selection of loaded data to produce sparse data set

        Keyword arguments:
        ratio -- the divisor of the length, e.g. 5 gives 1/5 of the data
        """
        indexes = np.random.randint(0, len(self.x_data),
                                    math.floor(len(self.x_data) / ratio))
        indexes.sort()

        rtn = AEData()
        rtn.x_data = self.x_data[indexes]
        rtn.y_data = self.y_data[indexes]
        rtn.t_data = self.t_data[indexes]
        rtn.timestamp = self.timestamp[indexes]

        return rtn

    def __repr__(self):
        return "{} total [x,y,t,ts]: [{}, {}, {}, {}]".format(len(self.x_data),
                                                              self.x_data, 
                                                              self.y_data,
                                                              self.t_data, 
                                                              self.timestamp)

    def __len__(self):
        return len(self.x_data)

    def interactive_animation(self, step=5000, limits=(0, 128), pause=0):
        """Present data in a graph that allows interaction"""
        plt.ion()
        fig = plt.figure(figsize=(6, 6))
        plt.show()
        axes = fig.add_subplot(111)

        start = 0
        end = step - 1
        while start < len(self.x_data):
            axes.clear()
            axes.scatter(self.x_data[start:end], self.y_data[start:end],
                         s=20, c=self.t_data[start:end], marker='o',
                         cmap='jet')
            axes.set_xlim(limits)
            axes.set_ylim(limits)
            start += step
            end += step
            plt.draw()
            time.sleep(pause)

    def downsample(self, new_dimensions=(16, 16)):
        """Reduces resolution to given dimensions

        Keyword arguments:
        new_dimensions -- tuple of new dimensions, default (16, 16)
        """
        # TODO
        # Make this cleaner
        assert self.dimensions[0] % new_dimensions[0] is 0
        assert self.dimensions[1] % new_dimensions[1] is 0

        rtn = AEData()
        rtn.timestamp = self.timestamp
        rtn.t_data = self.t_data
        rtn.x_data = np.floor(self.x_data / 
                              (self.dimensions[0] / new_dimensions[0]))
        rtn.y_data = np.floor(self.y_data / 
                              (self.dimensions[1] / new_dimensions[1]))

        return rtn

    def to_matrix(self, dim=(128, 128)):
        """Returns a matrix of the given dimensions containing AER data

        Keyword arguments:
        dim -- tuple of matrix dimensions, default (128, 128)
        """
        return make_matrix(self.x_data, self.y_data, self.t_data, dim=dim)

    def filter_events(self, evt_type):
        """Returns new AEData object with given event type removed

        Keyword arguments:
        evt_type -- string of event type, either 'ON' or 'OFF'"""
        if evt_type == 'ON':
            allow = 0
        elif evt_type == 'OFF':
            allow = 1
        else:
            print('Invalid event type for filter')
            return None
    
        rtn = AEData()
        
        for idx, _type in enumerate(self):
            if _type == allow:
                rtn.timestamp = np.append(rtn.timestamp, [self.timestamp[idx]])
                rtn.t_data = np.append(rtn.t_data, [allow])
                rtn.x_data = np.append(rtn.x_data, [self.x_data[idx]])
                rtn.y_data = np.append(rtn.y_data, [self.y_data[idx]])
        return rtn

    def merge_events(self):
        """Set all events to the same type, packaged in new AEData object"""
        rtn = copy.deepcopy(self)
        rtn.t_data = np.zeros(len(self.t_data))
        return rtn

    def take(self, n_elements):
        """Returns n_elements from data set by random selection"""
        if n_elements <= len(self):
            return self.make_sparse(len(self)/n_elements)
        else:
            print('Number of desired elements more than available')
            return None

    def take_v2(self, n_elements):
        """Takes n_elements elements with more even distribution"""
        if n_elements > len(self):
            print('Number of desired elements more than available')
            return None
        step = len(self)/n_elements
        temp = 0
        rtn = AEData()
        i = 0.0
        while i < len(self):
            numt = int(np.floor(i))
            if temp != numt:
                rtn.x_data = np.append(rtn.x_data, self.x_data[numt])
                rtn.y_data = np.append(rtn.y_data, self.y_data[numt])
                rtn.t_data = np.append(rtn.t_data, self.t_data[numt])
                rtn.timestamp = np.append(rtn.timestamp, self.timestamp[numt])
                temp = numt
            i += step
        return rtn

    def change_timescale(self, length, start=None):
        """Shifts and rescales timestamps, keeps starting point by default"""
        rtn = copy.deepcopy(self)
        _min = np.min(rtn.ts)
        if start is None:
            start = _min
        rtn.timestamp = np.floor(
            (rtn.timestamp - _min) /
            ((np.max(rtn.timestamp) - _min) / length) + start)
        return rtn


def make_matrix(x_data, y_data, t_data, dim=(128, 128)):
    """Form matrix from x, y, t data with given dimensions

    Keyword arguments:
    x_data -- x axis data
    y_data -- y axis data
    t_data -- t axis data
    dim -- tuple of matrix dimensions, default (128, 128)
    """
    image = np.zeros(dim)
    events = np.zeros(dim)

    for idx, _time in enumerate(t_data):
        image[y_data[idx]-1, x_data[idx]-1] -= _time - 0.5
        events[y_data[idx]-1, x_data[idx]-1] += 1

    # http://stackoverflow.com/questions/26248654/numpy-return-0-with-divide-by-zero
    np.seterr(divide='ignore', invalid='ignore')

    result = 0.5 + (image / events)
    result[events == 0] = 0.5
    return result


def create_pngs(data, prepend, path="", step=3000, dim=(128, 128)):
    """Create PNG files at given location with given dimensions"""
    if not os.path.exists(path):
        os.makedirs(path)

    idx = 0
    start = 0
    end = step - 1
    while start < len(data.x_data):
        image = make_matrix(data.x_data[start:end], data.y_data[start:end],
                            data.t_data[start:end], dim=dim)
        img_arr = (image*255).astype('uint8')
        im_data = Image.fromarray(img_arr)
        im_data.save(path + os.path.sep + prepend + ("%05d" % idx) + ".png")
        idx += 1

        start += step
        end += step


def concatenate(a_tuple):
    """Concatenate all tuple points into an AEData object"""
    rtn = AEData()
    n_points = len(a_tuple)
    rtn.x_data = np.concatenate(tuple([a_tuple[i].x_data for i in range(n_points)]))
    rtn.y_data = np.concatenate(tuple([a_tuple[i].y_data for i in range(n_points)]))
    rtn.t_data = np.concatenate(tuple([a_tuple[i].t_data for i in range(n_points)]))
    rtn.timestamp = np.concatenate(tuple([a_tuple[i].timestamp for i in range(n_points)]))
    return rtn
