"""Example of downsampling AEData from file"""

import pyaer

FILE = '/path/to/left_to_right_1.aedat'


def main():
    """Entry point of downsampling example"""

    lib = pyaer.AEFile(FILE, max_events=750001)
    data = pyaer.AEData(lib)

    sparse = data[300000:750000].make_sparse(64).downsample((16, 16))

    sparse.save_to_mat('downsample_left_to_right_1_1.mat')
    lib.save(sparse, 'downsample_left_to_right_1_1.aedat')

if __name__ == '__main__':
    main()
