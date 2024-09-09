'''
This module objective will be obtaining
characteristics of a specific dynamical
system in order to explain it.
'''

import numpy as np


def prepare_phase_space_values(
        x_lim: tuple,
        y_lim: tuple,
        n_arrows: complex,
        values: int,
        dx_nul: callable,
        dy_nul: callable,
        dyn_sys: callable,
        **params):
    '''
    This method will return all the parameters
    needed to plot a phase space of an autonomous
    dynamical system.

    :param x_lim: X-limit.
    :param y_lim: Y-limit.
    :param n_arrows: Number of arrows to generate along
    the meshgrid.
    :param values: Number of values to generate between
    the visual ranges mentioned.
    :param dx_nul: Function to compute the nulcline of dx.
    :param dy_nul: Function to compute the nulcline of dy.
    :param dyn_sys: Function to obtain the following state
    of our dynamical system (to create the arrows evolution
    of different trajectories).
    :param **params: Dynamic System parameters.

    :return list: We will return a list with all the
    needed values to plot a phase space. They will be
    in the following order:
    [x_window, y_window, ox, X, Y, dx_nul, dy_nul, dx, dy]
    Where:
        - x/y_window are the x/y limits of the plot.
        - ox is the OX-axis.
        - X/Y are the meshgrid needed to build the arrows.
        - dx/dy_nul are the nulclines.
        - dx/dy are the arrows directions. 
    '''
    # arrows meshgrid
    x = np.linspace(x_lim[0], x_lim[1], values)
    X, Y = np.mgrid[x_lim[0]:x_lim[1]:n_arrows,
                    y_lim[0]:y_lim[1]:n_arrows]

    # nulclines
    dx_nul_vals = dx_nul(x=x, **params)
    dy_nul_vals = dy_nul(x=x, **params)

    # arrows directions
    dx, dy = dyn_sys((X, Y), **params)

    return [x_lim, y_lim, x, X, Y,
            dx_nul_vals, dy_nul_vals,
            dx, dy]


def prepare_phase_space_values_around_point(
        x_visual_range: np.array,
        y_visual_range: np.array,
        point: np.array,
        n_arrows: complex,
        values: int,
        dx_nul: callable,
        dy_nul: callable,
        dyn_sys: callable,
        **params):
    '''
    This method will return all the parameters
    needed to plot a phase space of an autonomous
    dynamical system around a single point.

    :param x_visual_range: Visual X-limit around the
    critical point.
    :param y_visual_range: Visual Y-limit around the
    critical point.
    :param point: Critical point with the following
    coordinates: (x, y).
    :param n_arrows: Number of arrows to generate along
    the meshgrid.
    :param values: Number of values to generate between
    the visual ranges mentioned.
    :param dx_nul: Function to compute the nulcline of dx.
    :param dy_nul: Function to compute the nulcline of dy.
    :param dyn_sys: Function to obtain the following state
    of our dynamical system (to create the arrows evolution
    of different trajectories).
    :param **params: Dynamic System parameters.

    :return list: We will return a list with all the
    needed values to plot a phase space. They will be
    in the following order:
    [x_window, y_window, ox, X, Y, dx_nul, dy_nul, dx, dy]
    Where:
        - x/y_window are the x/y limits of the plot.
        - ox is the OX-axis.
        - X/Y are the meshgrid needed to build the arrows.
        - dx/dy_nul are the nulclines.
        - dx/dy are the arrows directions. 
    '''
    # visualization window around the critical point
    x_window = point[0] + x_visual_range
    y_window = point[1] + y_visual_range

    return prepare_phase_space_values(
        x_window, y_window, n_arrows, values,
        dx_nul, dy_nul, dyn_sys, **params)


def obtain_critical_points(
        coefs: list,
        equations: list):
    '''
    This method will obtain all the critical
    points of each equation passed through
    arguments, just for a 2D system!

    We well reject all the points with imaginary
    part either in the X or Y coordinates!

    It is necessary to leave all the equations
    in function of "x"!

    :param coefs: Coefficients that solves "x".
    :param equations: Equations from which obtain
    the "y" coordinates.
    :return tuple: A tuple with the solutions of "x"
    in a np.array(1 X len(x_sols)) and other array
    with the solutions for each equation, in other words:
    np.array(n_equations X len(x_sols)).
    '''
    roots = np.roots(coefs)
    x_sols = []
    for sol in roots:
        if np.isreal(sol).all():
            x_sols.append(sol)
    x_sols = np.asarray(x_sols)

    y_sols = np.zeros(
        (len(equations), len(x_sols)))

    for i, equation in enumerate(equations):
        y_sols[i, :] = equation(x_sols.real)

    return x_sols, y_sols


def prepare_bifurcation_plot(
        x_fun: callable,
        y_starts: list,
        y_stops: list,
        values: int):
    '''
    This method will prepare all the trajectories
    of the bifurcation diagram.

    :param x_fun: Function to generate "x" parameter
    in function of "y".
    :param y_starts: List with all the starting points
    to generate each trajectory.
    :param y_stops: List with all the ending points to
    generate each trajectory.
    :param values: Number of values to generate per
    trajectory.

    :return tuple: Tuple with two lists inside, one
    with all the "x" np.arrays (symbolizing all the "x"
    trajectories) and the other with all the "y" np.arrays.
    '''
    x_curves = []
    y_curves = []
    for y_start, y_stop in zip(y_starts, y_stops):
        y = np.linspace(y_start, y_stop, values)
        x = x_fun(y)

        x_curves.append(x)
        y_curves.append(y)

    return x_curves, y_curves
