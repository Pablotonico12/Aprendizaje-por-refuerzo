'''
This module will contain operations related
with the correct grid functioning or to obtain
certain coordinates of a cell position in the grid
taking into account that we are talking about a circular
grid (without edges).
'''

import numpy as np
import os
import datetime
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import imageio
from tqdm import tqdm


def get_upper_idx(
        grid: np.array,
        row: int, col: int):
    '''
    This method will return the
    upper index of our reference
    cell in the circular grid.

    :param grid: Our grid of neurons.
    :param row: Vertical reference
    from which we want to extract the
    corresponding index.
    :param col: Horizontal reference
    from which we want to extract the
    corresponding index.

    Example:
    ```
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

    row = 0
    col = 0

    get_upper_idx(matrix, row, col)
    ```

    Output:
    ```
    (2, 0)
    ```

    :return tuple: The upper index to
    our reference position `(row, col)`
    in the grid.
    '''
    height = grid.shape[1]
    return ((row - 1) % height, col)


def get_lower_idx(
        grid: np.array,
        row: int, col: int):
    '''
    This method will return the
    lower index of our reference
    cell in the circular grid.

    :param grid: Our grid of neurons.
    :param row: Vertical reference
    from which we want to extract the
    corresponding index.
    :param col: Horizontal reference
    from which we want to extract the
    corresponding index.

    Example:
    ```
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

    row = 0
    col = 0

    get_lower_idx(matrix, row, col)
    ```

    Output:
    ```
    (1, 0)
    ```

    :return tuple: The lower index to
    our reference position `(row, col)`
    in the grid.
    '''
    height = grid.shape[1]
    return ((row + 1) % height, col)


def get_right_idx(
        grid: np.array,
        row: int, col: int):
    '''
    This method will return the
    right index of our reference
    cell in the circular grid.

    :param grid: Our grid of neurons.
    :param row: Vertical reference
    from which we want to extract the
    corresponding index.
    :param col: Horizontal reference
    from which we want to extract the
    corresponding index.

    Example:
    ```
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

    row = 0
    col = 0

    get_right_idx(matrix, row, col)
    ```

    Output:
    ```
    (0, 1)
    ```

    :return tuple: The right index to
    our reference position `(row, col)`
    in the grid.
    '''
    width = grid.shape[0]
    return (row, (col + 1) % width)


def get_left_idx(
        grid: np.array,
        row: int, col: int):
    '''
    This method will return the
    left index of our reference
    cell in the circular grid.

    :param grid: Our grid of neurons.
    :param row: Vertical reference
    from which we want to extract the
    corresponding index.
    :param col: Horizontal reference
    from which we want to extract the
    corresponding index.

    Example:
    ```
    matrix = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

    row = 0
    col = 0

    get_left_idx(matrix, row, col)
    ```

    Output:
    ```
    (0, 2)
    ```

    :return tuple: The left index to
    our reference position `(row, col)`
    in the grid.
    '''
    width = grid.shape[0]
    return (row, (col - 1) % width)
