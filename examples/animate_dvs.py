"""Example of DVS animation"""

import pyaer


def main():
    """Entry point of animation example"""
    data = pyaer.AEData(pyaer.AEFile('/path/to/left_to_right_1.aedat',
                                     max_events=1000000))
    data = data.make_sparse(64).downsample((16, 16))
    data.interactive_animation(step=1000, limits=(0, 16))

if __name__ == '__main__':
    main()
