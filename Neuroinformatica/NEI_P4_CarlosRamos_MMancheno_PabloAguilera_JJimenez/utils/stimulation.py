'''
This module is expected to contain all the
necessary methods to introduce a stimuli into
a specific grid of neurons.
'''
import numpy as np
import warnings
from utils.ParameterEnum import ParameterEnum
from utils.grid_utils import (get_left_idx, get_right_idx,
                              get_lower_idx, get_upper_idx)
from utils.initialization import set_single_cell_with_array
from utils.common import transform_dict_params_into_array


def single_cell_stimulation(
        grid: np.array,
        row: int, col: int,
        stimuli: float):
    '''
    This method will stimulate a single
    cell specified by its coordinates
    in the grid.

    And it will add the current membrane
    potential into the "a(t-1)" field.

    :param grid: Our grid of neurons.
    :param row: Integer indicating the cell row.
    :param col: Integer indicating the cell column.
    :param stimuli: Real number with the stimulation
    to add to the current "a(t)".

    Example:
    ```
    matrix = np.zeros((2, 2, len(ParameterEnum)))

    values_dict = {
        ParameterEnum.A_T.name: 20, ParameterEnum.A_T_OLD.name: 1,
        ParameterEnum.P.name: 0.5, ParameterEnum.P_F.name: 0.9,
        ParameterEnum.G_1.name: 4, ParameterEnum.G_2.name: 5,
        ParameterEnum.G_3.name: 6, ParameterEnum.G_4.name: 7,
        ParameterEnum.C_P.name: 8, ParameterEnum.L.name: 9,
        ParameterEnum.L_SPIKE.name: 10, ParameterEnum.C_SPIKE.name: 11,
        ParameterEnum.C_NO_SPIKE.name: 12,
    }

    old_matrix = set_single_cell_with_dict(
        matrix, 0, 0, values_dict)
    new_matrix = single_cell_stimulation(
        old_matrix, 0, 0, 30)

    interest_idxs = (ParameterEnum.A_T.value,
                     ParameterEnum.A_T_OLD.value)

    old_matrix[0, 0, interest_idxs],\
        new_matrix[0, 0, interest_idxs]
    ```

    Output:
    ```
    (array([[20.],
            [ 1.]]),
     array([[50.],
            [20.]]))
    ```

    :return grid: We will return a copy of the
    grid with the stimulation already introduced.
    '''
    warnings.warn(
        "\nThis method does not introduce a stimulation " +
        "cluster like the mentioned in the paper of F. Borja,\n" +
        "P. Varona and N. Castellanos published in 2003, " +
        "it only adds an external potential to the specified cell.",
        category=DeprecationWarning, stacklevel=2)
    grid_copy = np.copy(grid)

    cell = grid_copy[row, col]
    cell[ParameterEnum.A_T_OLD.value] =\
        cell[ParameterEnum.A_T.value]
    cell[ParameterEnum.A_T.value] += stimuli

    return grid_copy


def create_cluster_with_array(
        grid: np.array,
        idx_ref: tuple,
        vert_ref: tuple,
        horiz_ref: tuple,
        values: np.array):
    '''
    This method will create a stimulation
    cluster with a specific set of common
    values described in `values`.

    :param grid: Our grid of neurons.
    :param idx_ref: Tuple with the reference
    index to the cluster center, the structure
    should be: `(row_index, col_index)`.
    :param vert_ref: Tuple with the number of
    cells above and below of the reference
    cell, the structure should be:
    `(n_above_cells, n_below_cells)`
    :param horiz_ref: Tuple with the number of
    cells to the left and to the right of the
    reference cell, the structure should be:
    `(n_left_cells, n_right_cells)`
    :param values_dict: Array with all the values.

    Example:
    ```
    matrix = np.zeros((8, 8, len(ParameterEnum)))

    values_dict = {
        ParameterEnum.A_T.name: 20, ParameterEnum.A_T_OLD.name: 1,
        ParameterEnum.P.name: 0.5, ParameterEnum.P_F.name: 0.9,
        ParameterEnum.G_1.name: 4, ParameterEnum.G_2.name: 5,
        ParameterEnum.G_3.name: 6, ParameterEnum.G_4.name: 7,
        ParameterEnum.C_P.name: 8, ParameterEnum.L.name: 9,
        ParameterEnum.L_SPIKE.name: 10, ParameterEnum.C_SPIKE.name: 11,
        ParameterEnum.C_NO_SPIKE.name: 12,
    }
    values = transform_dict_params_into_array(values_dict)

    # We want a 5x5 cluster around [4, 4]
    idx_ref = (4, 4)
    vert_ref = (2, 2)
    horiz_ref = (2, 2)

    new_matrix = create_cluster_with_dict(
        matrix, idx_ref, vert_ref, horiz_ref, values)

    # We repeat it 5 times because it is a 5x5 cluster
    # (the reference cell + 2 cells above + 2 cells below)
    # And it is the same for the right & left cells of
    # the cluster.
    expected_values =\
        np.repeat(values.reshape(1, len(ParameterEnum)), 5, axis=0)

    np.all(new_matrix[2:7, 4] == expected_values),\
        np.all(new_matrix[4, 2:7] == expected_values),\
            np.all(new_matrix[0:4, 0:4] == 0),\
                np.all(new_matrix[5:9, 5:9] == 0)
    ```

    Output:
    ```
    (True, True, True, True)
    ```

    :return grid: We will return a copy of the
    grid with the cluster created.
    '''
    new_grid = np.copy(grid)

    row, col = idx_ref
    new_grid = set_single_cell_with_array(
        new_grid, row, col, values)

    row_upper = row_lower = row
    col_left = col_right = col

    for _ in range(vert_ref[0]):
        row_upper, _ = get_upper_idx(
            new_grid, row_upper, col)
        new_grid = set_single_cell_with_array(
            new_grid, row_upper, col, values)

    for _ in range(vert_ref[1]):
        row_lower, _ = get_lower_idx(
            new_grid, row_lower, col)
        new_grid = set_single_cell_with_array(
            new_grid, row_lower, col, values)

    for _ in range(horiz_ref[0]):
        _, col_left = get_left_idx(
            new_grid, row, col_left)
        new_grid = set_single_cell_with_array(
            new_grid, row, col_left, values)

    for _ in range(horiz_ref[1]):
        _, col_right = get_right_idx(
            new_grid, row, col_right)
        new_grid = set_single_cell_with_array(
            new_grid, row, col_right, values)

    return new_grid


def create_cluster_with_dict(
        grid: np.array,
        idx_ref: tuple,
        vert_ref: tuple,
        horiz_ref: tuple,
        values_dict: dict):
    '''
    This method will create a stimulation
    cluster with a specific set of common
    values described in `values_dict`.

    :param grid: Our grid of neurons.
    :param idx_ref: Tuple with the reference
    index to the cluster center, the structure
    should be: `(row_index, col_index)`.
    :param vert_ref: Tuple with the number of
    cells above and below of the reference
    cell, the structure should be:
    `(n_above_cells, n_below_cells)`
    :param horiz_ref: Tuple with the number of
    cells to the left and to the right of the
    reference cell, the structure should be:
    `(n_left_cells, n_right_cells)`
    :param values_dict: Dictionary with all the values
    to set.

    Example:
    ```
    matrix = np.zeros((8, 8, len(ParameterEnum)))

    values_dict = {
        ParameterEnum.A_T.name: 20, ParameterEnum.A_T_OLD.name: 1,
        ParameterEnum.P.name: 0.5, ParameterEnum.P_F.name: 0.9,
        ParameterEnum.G_1.name: 4, ParameterEnum.G_2.name: 5,
        ParameterEnum.G_3.name: 6, ParameterEnum.G_4.name: 7,
        ParameterEnum.C_P.name: 8, ParameterEnum.L.name: 9,
        ParameterEnum.L_SPIKE.name: 10, ParameterEnum.C_SPIKE.name: 11,
        ParameterEnum.C_NO_SPIKE.name: 12,
    }

    # We want a 5x5 cluster around [4, 4]
    idx_ref = (4, 4)
    vert_ref = (2, 2)
    horiz_ref = (2, 2)

    new_matrix = create_cluster_with_dict(
        matrix, idx_ref, vert_ref, horiz_ref, values_dict)

    values = transform_dict_params_into_array(values_dict)

    # We repeat it 5 times because it is a 5x5 cluster
    # (the reference cell + 2 cells above + 2 cells below)
    # And it is the same for the right & left cells of
    # the cluster.
    expected_values =\
        np.repeat(values.reshape(1, len(ParameterEnum)), 5, axis=0)

    np.all(new_matrix[2:7, 4] == expected_values),\
        np.all(new_matrix[4, 2:7] == expected_values),\
            np.all(new_matrix[0:4, 0:4] == 0),\
                np.all(new_matrix[5:9, 5:9] == 0)
    ```

    Output:
    ```
    (True, True, True, True)
    ```

    :return grid: We will return a copy of the
    grid with the cluster created.
    '''
    return create_cluster_with_array(
        grid, idx_ref, vert_ref, horiz_ref,
        transform_dict_params_into_array(values_dict))
