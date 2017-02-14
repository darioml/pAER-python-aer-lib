"""Load data from four directions in separate files"""

from __future__ import print_function
import numpy as np
import pyaer

BASE_DIR = '/path/to/some/dir/'
FILE1 = 'left_to_right_1.aedat'
FILE2 = 'right_to_left_1.aedat'
FILE3 = 'top_to_bottom_1.aedat'
FILE4 = 'bottom_to_top_1.aedat'

# Each ball movement should be .5s long
ANIMATION_TIME = 0.2

# 3,280 events per second for 16*16 is reasonable
# for ball movement (might be even too high!)
NUM_EVENTS_P_S = 3280


def get_data(file, min, max, animation_time=ANIMATION_TIME,
             num_events=NUM_EVENTS_P_S*ANIMATION_TIME, offset=0):
    """
    Helper function to read a file.

    Given (min,max) which are data ranges for extraction, this will return a
    cropped and suitable sparse output.
    """
    aefile = pyaer.AEFile(file, max_events=max+1)
    aedata = pyaer.AEData(aefile)
    print("Points: {}, Time: {:0.2f}, Sparsity: {}".format(
        len(aefile.data),
        (aefile.timestamp[-1] - aefile.timestamp[0]) / 1000000,
        np.floor(len(aefile.data)/num_events)))

    sparse = aedata[min:max].make_sparse(np.floor(len(aefile.data)/num_events))

    actual_time = (sparse.ts[-1]-sparse.ts[0])/1000000
    scale = actual_time/animation_time
    sparse.ts = ((offset * 1000000) +
                 np.round((sparse.timestamp - sparse.timestamp[0]) / scale))

    return sparse


def main():
    """Entry point of example"""
    # Loop through all files - indexes are extrapolated.
    data1 = get_data(FILE1, 300000, 750000, offset=0*ANIMATION_TIME)
    data2 = get_data(FILE2, 300000, 600000, offset=1*ANIMATION_TIME)
    data3 = get_data(FILE3, 85000, 140000, offset=2*ANIMATION_TIME)
    data4 = get_data(FILE4, 65200, 131800, offset=3*ANIMATION_TIME)

    # Need to pre-load a file, to get the correct headers when writing!
    lib = pyaer.AEFile(FILE1, max_events=1)

    final = pyaer.concatenate((data1, data2, data3, data4))
    final_16 = final.downsample((16, 16))

    lib.save(final, 'test.aedat')
    lib.save(final_16, 'test_16.aedat')

    data1.save_to_mat('test_1.mat')
    data2.save_to_mat('test_2.mat')
    data3.save_to_mat('test_3.mat')
    data4.save_to_mat('test_4.mat')


if __name__ == '__main__':
    main()
