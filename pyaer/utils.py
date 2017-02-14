"""
Author: Dario ML, bdevans, Michael Hart
Program: pyaer/utils.py
Description: Miscellaneous utility functions for PyAER
"""

import os
from PIL import Image
import numpy as np
from pyaer import AEData


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
    rtn.x_data = np.concatenate(
        tuple([a_tuple[i].x_data for i in range(n_points)]))
    rtn.y_data = np.concatenate(
        tuple([a_tuple[i].y_data for i in range(n_points)]))
    rtn.t_data = np.concatenate(
        tuple([a_tuple[i].t_data for i in range(n_points)]))
    rtn.timestamp = np.concatenate(
        tuple([a_tuple[i].timestamp for i in range(n_points)]))
    return rtn
