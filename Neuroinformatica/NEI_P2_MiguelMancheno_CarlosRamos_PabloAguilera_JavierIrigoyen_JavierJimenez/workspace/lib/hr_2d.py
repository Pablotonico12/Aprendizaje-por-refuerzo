'''
In this module we will define all the methods
to simulate a Hindmarsh-Rose 2D system.
'''

from collections import deque
import numpy as np

# Default values
A = 1
B = 2.6
C = 1
D = 5

I_0 = 1
I_INF = 2

Z = 0

STABLE_STATE = \
    (-0.8610800560651974, -2.7984503338411115)
SILENT_STATE = \
    (0.6, 1)

T_START, T_END, T_STEP = 0, 150, 1e-3


def I(
        t: float,
        pulse_times: deque,
        i_0: float,
        i_inf: float):
    '''
    This method will choose whether to return
    I_0 or I_inf dependig on pulse_times queue.

    :param t: Current time.
    :param pulse_times: Times to throw a pulse in
    a stack class.
    :param i_0: I_0 value (it is returned when
    there is a pulse).
    :param i_inf: I_inf value (it is returned
    when there is not a pulse).
    :return float: It will return I_0 or I_inf
    '''
    if len(pulse_times) == 0:
        return i_inf

    t0, tf = pulse_times.popleft()
    if t0 <= t <= tf:
        pulse_times.appendleft((t0, tf))
        return i_0
    elif t < t0:
        pulse_times.appendleft((t0, tf))

    return i_inf


def dx_dt(state: tuple, t: float, a: float,
          b: float, z: float, **kwargs):
    '''
    dx/dt integration.

    :param state: Current state of the system.
    :param t: Current time.
    :param a: A parameter.
    :param b: B parameter.
    :param z: z parameter.
    :return float: The system's integration.
    '''
    x, y = state
    return y - a * (x*x*x) + b * (x * x) + I(t, **kwargs) - z


def dy_dt(state: tuple, t: float, c: float, d: float):
    '''
    dy/dt integration.

    :param state: Current state of the system.
    :param t: Current time.
    :param c: C parameter.
    :param d: D parameter.
    :return float: The system's integration.
    '''
    x, y = state
    return c - d * (x*x) - y


def hr_dt(state: tuple, t: float, a: float, b: float, c: float,
          d: float, pulse_times: deque, i_0: float, i_inf: float,
          z: float, **kwargs):
    '''
    Entire Hindmarsh-Rose integration.

    :param state: Current state of the system.
    :param t: Current time.
    :param a: A parameter.
    :param b: B parameter.
    :param c: C parameter.
    :param d: D parameter.
    :param pulse_times: Times to throw a pulse in
    a stack class.
    :param i_0: I_0 value (it is returned when
    there is a pulse).
    :param i_inf: I_inf value (it is returned
    when there is not a pulse).
    :param z: z parameter.
    :return tuple: A tuple with the dx & dy
    integrations.
    '''
    dx_params =\
        {'a': a, 'b': b, 'pulse_times': pulse_times,
         'i_0': i_0, 'i_inf': i_inf, 'z': z}
    dy_params =\
        {'c': c, 'd': d}

    return (dx_dt(state, t, **dx_params),
            dy_dt(state, t, **dy_params))


def dx_nul(a: float, x: np.array, b: float,
           i: float, z: float, **kwargs):
    '''
    This method will obtain the dx/dt nulcline.

    :param a: A parameter.
    :param x: OX linespace.
    :param b: B parameter.
    :param i: I_inf/I_0 parameter
    :param z: z parameter.
    :return np.array: A series of float
    values with the dx/dt nulcline
    '''
    x_square = x*x
    return a*x_square*x - b*x_square - i + z


def dy_nul(c: float, d: float, x: np.array, **kwargs):
    '''
    This method will obtain the dy/dt nulcline.

    :param c: C parameter.
    :param d: D parameter.
    :param x: OX linespace.
    :return np.array: A series of float
    values with the dy/dt nulcline
    '''
    return c - d*(x*x)
