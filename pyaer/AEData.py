"""
Author: Dario ML, bdevans, Michael Hart
Program: pyaer/AEData.py
Description: AEData class definition for PyAER
"""


import numpy as np
from pyaer import AEFile


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
