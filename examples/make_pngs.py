"""Example to create PNG files from data"""

from os import listdir
from os.path import isfile, join
import pyaer

MYPATH = '/path/to/fyp-aedata-matlab'
ONLYFILES = [f for f in listdir(MYPATH)
             if isfile(join(MYPATH, f)) and f.endswith('.aedat')]


def main():
    """Entry point of PNG file example"""

    for _file in ONLYFILES:
        aef = pyaer.AEFile('path/to/fyp-aedata-matlab/' + str(_file))
        aed = pyaer.AEData(aef).downsample((16, 16))

        pyaer.create_pngs(aed, '16x16_' + str(_file) + '_',
                          path='testing_something', step=3000, dim=(16, 16))

if __name__ == '__main__':
    main()
