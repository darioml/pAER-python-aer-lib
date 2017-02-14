"""
Author: Dario ML, bdevans, Michael Hart
Program: pyaer/AEFile.py
Description: PyAER AEFile class definition
"""

import math
import numpy as np
from pyaer import AEData


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
                    aef.write(
                        hex(int(data[i]))[2:].zfill(8).decode('hex'))
                    aef.write(
                        hex(int(timestamp[i]))[2:].zfill(8).decode('hex'))

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
