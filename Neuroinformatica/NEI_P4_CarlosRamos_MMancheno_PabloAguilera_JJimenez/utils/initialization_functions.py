'''
This module is expected to contain all the
functions used by the "set_all_cells" method
contained in "initialization.py".

All the methods contained in this file will
write a specific set of values at each cell.

The common structure should be:
    ```
    def my_fun(i: int, j: int, ...):
        ...
    ```

Where `i` is the cell row index, and `j`
the cell column index.
'''

from utils.common import transform_dict_params_into_array


def identity_init_fn(
        i: int, j: int, val: float):
    '''
    This method will write a constant
    value `val` in all the cell parameters
    and in each cell of the grid.

    :param i: Cell row index.
    :param j: Cell column index.
    :param val: Constant value.

    Example:
    ```
    width = 2
    height = 2

    matrix = np.zeros(
        (width, height, len(ParameterEnum)))

    set_all_cells(matrix, identity_fn, val=0.4)
    ```

    Output:
    ```
    array([[[0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
         0.4],
        [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
         0.4]],

       [[0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
         0.4],
        [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
         0.4]]])
    ```

    :return val: It will return the same
    value passed as parameter.
    '''
    return val


def dict_init_fn(
        i: int, j: int, _dict: dict):
    '''
    This method will put the same `_dict` in
    every cell.

    :param i: Cell row index.
    :param j: Cell column index.
    :param _dict: Dictionary to write.

    Example:
    ```
    width = 2
    height = 2

    values_dict = {
            ParameterEnum.A_T.name: 0, ParameterEnum.A_T_OLD.name: 1,
            ParameterEnum.P.name: 0.5, ParameterEnum.P_F.name: 0.9,
            ParameterEnum.G_1.name: 4, ParameterEnum.G_2.name: 5,
            ParameterEnum.G_3.name: 6, ParameterEnum.G_4.name: 7,
            ParameterEnum.C_P.name: 8, ParameterEnum.L.name: 9,
            ParameterEnum.L_SPIKE.name: 10, ParameterEnum.C_SPIKE.name: 11,
            ParameterEnum.C_NO_SPIKE.name: 12,
        }


    matrix = np.zeros(
        (width, height, len(ParameterEnum)))

    set_all_cells(matrix, dict_init_fn, _dict=values_dict)
    ```

    Output:
    ```
    array([[[ 0. ,  1. ,  0.5,  0.9,  4. ,  5. ,  6. ,  7. ,  8. ,  9. ,
         10. , 11. , 12. ],
        [ 0. ,  1. ,  0.5,  0.9,  4. ,  5. ,  6. ,  7. ,  8. ,  9. ,
         10. , 11. , 12. ]],

       [[ 0. ,  1. ,  0.5,  0.9,  4. ,  5. ,  6. ,  7. ,  8. ,  9. ,
         10. , 11. , 12. ],
        [ 0. ,  1. ,  0.5,  0.9,  4. ,  5. ,  6. ,  7. ,  8. ,  9. ,
         10. , 11. , 12. ]]])
    ```

    :return np.array: Array with the `_dict`
    parameters transformed.
    '''
    return transform_dict_params_into_array(_dict)
