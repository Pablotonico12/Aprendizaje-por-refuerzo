'''
This module will contain all those
functions used by multiple modules
which do not refer to any other custom
module!
'''
import numpy as np
from utils.ParameterEnum import ParameterEnum


def transform_dict_params_into_array(_dict: dict):
    '''
    This method will transform a dictionary
    of parameters into a numpy array containing
    these parameters in order.

    :param _dict: Dictionary with all the values.
    It is necessary for this parameter to have
    the `ParameterEnum` names as keys.

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

    transform_dict_params_into_array(values_dict)
    ```

    Output:
    ```
    array([ 0. ,  1. ,  0.5,  0.9,  4. ,  5. ,  6. ,  7. ,  8. ,  9. , 10. ,
       11. , 12. ])
    ```

    :return np.array: Array with all the values.
    '''
    values = np.zeros(len(_dict))
    for key, val in _dict.items():
        values[ParameterEnum[key].value] = val

    return values
