'''
This module is expected to contain all the
necessary methods to initialize a grid of
neurons or setting up a specific cell value.
'''
from typing import List
from utils.ParameterEnum import ParameterEnum
from utils.common import transform_dict_params_into_array
import numpy as np


def IS_PROB(x): return 0 <= x <= 1


def is_a_valid_cell(
        cell_values: np.array,
        criteria: List[callable]):
    '''
    This method will check whether the cell
    contain valid values or not, this method
    allows multiple criteria.

    :param cell_values: Array with all the
    cell actual values.
    :param criteria: List of criteria where each
    element is a predicate (a function which returns
    a boolean).

    Example:
    ```
    N_COMP = 13
    values = np.linspace(0, 10, N_COMP)
    is_a_valid_cell(values, [lambda x: x[0] == 0,
                             lambda x: x[1] == 23])
    ```

    Output:
    ```
    False
    ```

    :return bool: True if all criteria are met,
    false otherwise.
    '''
    for crit in criteria:
        if not crit(cell_values):
            return False

    return True


def check_cell_probabilities(cell: np.array):
    '''
    This method will check whether a cell has
    probabilities in it or not. If it has not
    a probability, it will throw an Exception.

    :param cell: Array with all the cell values.

    Example:
    ```
    N_COMP = 13
    values = np.linspace(1, 2, N_COMP)
    check_cell_probabilities(values)
    ```

    Output:
    ```
    ...
    Exception: (utils/common.py: check_cell_probabilities):
    You did not entered a probability at P or P_F
    ```

    :return None:
    '''
    if not is_a_valid_cell(
            cell, [lambda x: IS_PROB(x[ParameterEnum.P.value])
                   and IS_PROB(x[ParameterEnum.P_F.value])]):
        raise Exception(
            "(utils/common.py: check_cell_probabilities): " +
            "You did not entered a probability at {} or {}".
            format(ParameterEnum.P.name, ParameterEnum.P_F.name))


def check_values_len(cell: np.array):
    '''
    This method will check whether our
    cell number of values is equal to the
    number of values defined in `ParameterEnum`.

    If it has not the expected number of values
    (len(ParameterEnum)), then it will raise an
    Exception.

    :param cell: Array with all the values.

    Example:
    ```
    (imagine len(ParameterEnum) = 13)
    check_values_len(np.zeros(12))
    ```

    Output:
    ```
    ...
    Exception: (utils/common.py: check_values_len):
    The expected number of values to receive was 13,
    but 12 were given
    ```

    :return None:
    '''
    if len(ParameterEnum) != len(cell):
        raise Exception(
            "(utils/common.py: check_values_len): The expected" +
            " number of values to receive was {}, but {} were given".
            format(len(ParameterEnum), len(cell)))


def set_single_cell_with_array(
        grid: np.array,
        row: int, col: int,
        values: np.array):
    '''
    This method will initialize a single cell
    located in a specific "row" and "col".

    :param grid: Our grid of neurons.
    :param row: Integer indicating the cell row.
    :param col: Integer indicating the cell column.
    :param values: Array with all the values.

    Example:
    ```
    N_COMP = 13
    width, height = 50, 50
    matrix = np.zeros(
        (width, height, N_COMP))

    values = np.linspace(0, 1, N_COMP)
    new_matrix = set_single_cell_with_array(
        matrix, 0, 0, values)
    np.all(new_matrix[0, 0] == values)
    ```

    Output:
    ```
    True
    ```

    :return grid: We will return a copy of the
    grid with the value already set.
    '''
    check_values_len(values)
    check_cell_probabilities(values)
    grid_copy = np.copy(grid)

    grid_copy[row, col] = values
    return grid_copy


def set_single_cell_with_dict(
        grid: np.array,
        row: int, col: int,
        values_dict: dict):
    '''
    This method will initialize a single cell
    located in a specific "row" and "col".

    :param grid: Our grid of neurons.
    :param row: Integer indicating the cell row.
    :param col: Integer indicating the cell column.
    :param values_dict: Dictionary with all the values
    to set.

    Example:
    ```
    values_dict = {
        ParameterEnum.A_T.name: 0, ParameterEnum.A_T_OLD.name: 1,
        ParameterEnum.P.name: 0.5, ParameterEnum.P_F.name: 0.9,
        ParameterEnum.G_1.name: 4, ParameterEnum.G_2.name: 5,
        ParameterEnum.G_3.name: 6, ParameterEnum.G_4.name: 7,
        ParameterEnum.C_P.name: 8, ParameterEnum.L.name: 9,
        ParameterEnum.L_SPIKE.name: 10, ParameterEnum.C_SPIKE.name: 11,
        ParameterEnum.C_NO_SPIKE.name: 12,
    }

    new_matrix = set_single_cell_with_dict(
        matrix, 0, 0, values_dict)
    new_matrix[0, 0]
    ```

    Output:
    ```
    array([ 0. ,  1. ,  0.5,  0.9,  4. ,  5. ,
      6. ,  7. ,  8. ,  9. , 10. , 11. , 12. ])
    ```

    :return grid: We will return a copy of the
    grid with the value already set.
    '''
    return set_single_cell_with_array(
        grid, row, col, transform_dict_params_into_array(values_dict))


def set_all_cells(grid: np.array, fun: callable, **kwargs):
    '''
    This method will assign values given by "fun"
    for each cell.

    The extra parameters will be passed through "fun".

    :param grid: The full neuron grid.
    :param fun: A function which will receive
    two parameters:
        - i: int -> The row index.
        - j: int -> The column index.

    Example:
    ```
    width = 2
    height = 2

    matrix = np.zeros(
        (width, height, len(ParameterEnum)))

    set_all_cells(matrix, lambda _, __, k: k, k=0.3)
    ```

    Output:
    ```
    array([[[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
         0.3],
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
         0.3]],

       [[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
         0.3],
        [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
         0.3]]])
    ```

    :return np.array: A copy of the grid with
    the new initialization.
    '''
    grid_copy = np.copy(grid)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            grid_copy[i, j] = fun(i, j, **kwargs)
            check_cell_probabilities(
                grid_copy[i, j])

    return grid_copy
