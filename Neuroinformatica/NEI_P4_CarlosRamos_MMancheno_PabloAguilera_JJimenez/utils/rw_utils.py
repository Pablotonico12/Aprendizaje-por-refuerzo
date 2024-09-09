'''
This module will contain all the operations related
with the correct random walk functioning.
'''

from utils.ParameterEnum import ParameterEnum
import datetime
import numpy as np
import random
import pickle
import os


# Status indices
SUBTHRESHOLD = 0
SPIKE = 1
POST_SPIKE = 2
FAILED_SPIKE = 3


def get_neuron(
        grid: np.array,
        x: int,
        y: int):
    '''
    This method will return a pointer
    to the np.array of the neuron
    desired.

    :param grid: Grid of neurons.
    :param x: Row index of the requested
    neuron.
    :param y: Column index of the requested
    neuron.

    Example:
    ```
    matrix = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    get_neuron(matrix, 1, 1)
    ```

    Output:
    ```
    5
    ```

    :return np.array: Array with all
    the values of the neuron selected.
    '''
    return grid[x, y]


def get_up(
        grid: np.array,
        x: int,
        y: int):
    '''
    This method will return a pointer
    to the np.array of the upside neuron
    to the current to emulate a circular
    grid.

    :param grid: Grid of neurons.
    :param x: Row index of the current
    neuron.
    :param y: Column index of the current
    neuron.

    Example:
    ```
    matrix = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    get_up(matrix, 0, 0)
    ```

    Output:
    ```
    7
    ```

    :return np.array: Array with all
    the values of the neuron selected.
    '''
    x_neighbor = (x - 1) % grid.shape[0]
    return get_neuron(grid, x_neighbor, y)


def get_down(
        grid: np.array,
        x: int,
        y: int):
    '''
    This method will return a pointer
    to the np.array of the downside neuron
    to the current to emulate a circular
    grid.

    :param grid: Grid of neurons.
    :param x: Row index of the current
    neuron.
    :param y: Column index of the current
    neuron.

    Example:
    ```
    matrix = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    get_down(matrix, 2, 1)
    ```

    Output:
    ```
    2
    ```

    :return np.array: Array with all
    the values of the neuron selected.
    '''
    x_neighbor = (x + 1) % grid.shape[0]
    return get_neuron(grid, x_neighbor, y)


def get_right(
        grid: np.array,
        x: int,
        y: int):
    '''
    This method will return a pointer
    to the np.array of the rightside neuron
    to the current to emulate a circular
    grid.

    :param grid: Grid of neurons.
    :param x: Row index of the current
    neuron.
    :param y: Column index of the current
    neuron.

    Example:
    ```
    matrix = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    get_right(matrix, 1, 2)
    ```

    Output:
    ```
    4
    ```

    :return np.array: Array with all
    the values of the neuron selected.
    '''
    y_neighbor = (y + 1) % grid.shape[1]
    return get_neuron(grid, x, y_neighbor)


def get_left(
        grid: np.array,
        x: int,
        y: int):
    '''
    This method will return a pointer
    to the np.array of the leftside neuron
    to the current to emulate a circular
    grid.

    :param grid: Grid of neurons.
    :param x: Row index of the current
    neuron.
    :param y: Column index of the current
    neuron.

    Example:
    ```
    matrix = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

    get_left(matrix, 0, 0)
    ```

    Output:
    ```
    3
    ```

    :return np.array: Array with all
    the values of the neuron selected.
    '''
    y_neighbor = (y - 1) % grid.shape[1]
    return get_neuron(grid, x, y_neighbor)


def random_walk_step(
        grid: np.array,
        x: int,
        y: int):
    '''
    This method will advance a step in
    time for a specific neuron.

    As the changes are done with the
    cell pointer there is no need to
    return nothing.

    :param grid: Grid of neurons.
    :param x: Row index of the current
    neuron.
    :param y: Column index of the current
    neuron.

    Example:
    ```
    for t in range(steps):
        # I scroll through the entire matrix
        # to update the state of the neurons.
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                # Execute step
                random_walk_step(grid, i, j)
    ```

    Output:
    ```
    <Each cell iterated will change its values
     implicitly (it works with pointers) depending
     on its neighboors>
    ```
    '''
    neuron = get_neuron(grid, x, y)

    # Update the AOLD value of the neuron
    neuron[ParameterEnum.A_T_OLD.value] =\
        neuron[ParameterEnum.A_T.value]

    # Update A according the neuron state
    if neuron[ParameterEnum.STAT.value] == SUBTHRESHOLD:
        if (random.random() >=
                neuron[ParameterEnum.P_F.value]):
            # neuron[ParameterEnum.C_P.value]
            neuron[ParameterEnum.A_T.value] += 1

    elif neuron[ParameterEnum.STAT.value] == SPIKE:
        neuron[ParameterEnum.A_T.value] +=\
            neuron[ParameterEnum.L_SPIKE.value]
        neuron[ParameterEnum.STAT.value] = POST_SPIKE

    elif neuron[ParameterEnum.STAT.value] == POST_SPIKE:
        if (random.random() >=
                neuron[ParameterEnum.P_F.value]):
            neuron[ParameterEnum.A_T.value] -=\
                neuron[ParameterEnum.C_SPIKE.value]

    elif neuron[ParameterEnum.STAT.value] == FAILED_SPIKE:
        if (random.random() >=
                neuron[ParameterEnum.P_F.value]):
            neuron[ParameterEnum.A_T.value] -=\
                neuron[ParameterEnum.C_NO_SPIKE.value]

    # Calculate de neighbor term
    up_neighbor = get_up(grid, x, y)
    a_neighbor = neuron[ParameterEnum.G_1.value] *\
        (up_neighbor[ParameterEnum.A_T_OLD.value] -
            neuron[ParameterEnum.A_T_OLD.value])

    down_neighbor = get_down(grid, x, y)
    a_neighbor += neuron[ParameterEnum.G_2.value] *\
        (down_neighbor[ParameterEnum.A_T_OLD.value] -
            neuron[ParameterEnum.A_T_OLD.value])

    left_neighbor = get_left(grid, x, y)
    a_neighbor += neuron[ParameterEnum.G_3.value] *\
        (left_neighbor[ParameterEnum.A_T_OLD.value] -
            neuron[ParameterEnum.A_T_OLD.value])

    right_neighbor = get_right(grid, x, y)
    a_neighbor += neuron[ParameterEnum.G_4.value] *\
        (right_neighbor[ParameterEnum.A_T_OLD.value] -
            neuron[ParameterEnum.A_T_OLD.value])

    # And add it to the peviously calculated value for A
    if neuron[ParameterEnum.STAT.value] != POST_SPIKE and\
            neuron[ParameterEnum.STAT.value] != FAILED_SPIKE:
        neuron[ParameterEnum.A_T.value] += a_neighbor

    # Do status checks after the final A value calculation
    if neuron[ParameterEnum.A_T.value] >= neuron[ParameterEnum.L.value] and\
            (neuron[ParameterEnum.STAT.value] != POST_SPIKE and
                neuron[ParameterEnum.STAT.value] != FAILED_SPIKE):
        # We use P_F to check if we have a spike or a failed spike on the other hand
        if (random.random() >= neuron[ParameterEnum.P_F.value]):
            neuron[ParameterEnum.STAT.value] = SPIKE
        else:
            neuron[ParameterEnum.STAT.value] = FAILED_SPIKE

    elif neuron[ParameterEnum.A_T.value] <= 0:
        neuron[ParameterEnum.A_T.value] = 0
        neuron[ParameterEnum.STAT.value] = SUBTHRESHOLD


def random_walk(
        grid: np.array,
        steps: int):
    '''
    This method will iterate `steps` times
    our grid calling the `random_walk_step`
    method.

    :param grid: Grid of neurons.
    :param steps: Number of steps to iterate.

    Example:
    ```
    grid_first_walk = random_walk(
        grid_first_sim, STEPS)
    ```

    Output:
    ```
    <A 3D array with described in the return section>
    ```

    :return grid_walk: This method will return
    a 3D-array where the first dimension represents
    the time, and the last two ones represent the
    grid rows and columns, respectively.
    '''
    # Add the first step (t0) to the walk
    grid_walk = np.zeros(
        (steps, grid.shape[0], grid.shape[1]))

    for t in range(steps):
        # I scroll through the entire matrix
        # to update the state of the neurons.
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                # Execute step
                random_walk_step(grid, i, j)

        # We only store the membrane potential of each cell!
        grid_walk[t] = grid[:, :, ParameterEnum.A_T.value[0]]

    return grid_walk


def save_grid_walk(
        grid_walk: np.array,
        name: str,
        path: str = '',
        add_timestamp: bool = True):
    '''
    This method will store the current `grid_walk` in a
    pickle file to avoid future executions.

    :return grid_walk: This parameter is a 3D-array
    where the first dimension represents the time,
    and the last two ones represent the
    grid rows and columns, respectively.
    :param name: Name of the file to be generated.
    :param path: Path where the file will be stored,
    defaults to ''
    :param add_timestamp: This parameter will add to the
    file name a timestamp, defaults to True

    Example:
    ```
    save_grid_walk(
        random_walk, 'random_walk', add_timestamp=False)
    ```

    Output:
    ```
    <A new file named 'random_walk'>
    ```
    '''
    if add_timestamp:
        name += '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    file_path = os.path.join(path, name + '.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(grid_walk, file)


def load_grid_walk(
        name: str,
        path: str = ''):
    '''
    This method will load a specific pickle
    file via parameters.

    :param name: Name of the gif to be generated.
    :param path: Path the gif will be stored, defaults to ''

    Example:
    ```
    save_grid_walk(
        random_walk, 'random_walk', add_timestamp=False)
    load_grid_walk('random_walk.pkl')
    ```

    Output:
    ```
    <The same `random_walk` array>
    ```

    :return np.array: This method will return our desired
    `grid_walk`.
    '''
    file_path = os.path.join(path, name)
    with open(file_path, 'rb') as file:
        return pickle.load(file)
