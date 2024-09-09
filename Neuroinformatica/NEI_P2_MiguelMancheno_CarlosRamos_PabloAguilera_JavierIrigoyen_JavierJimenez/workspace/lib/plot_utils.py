'''
This module will add some methods responsible
of the graphics visualization.
'''

import matplotlib.pyplot as plt
import numpy as np

# time plots
TIME_TITLE = "Membrane Potential Behaviour"

TIME_OX_LABEL = "Time (ms)"
TIME_OY_LABEL = "Membrane potential (mV)"

# phase space plots
OX_LABEL = "Membrane potential (mV)"
OY_LABEL = "Recovery variable"
TITLE = "Membrane Potential vs Recovery variable - Dynamics"

X_NUL_TITLE = "X nulcline"
Y_NUL_TITLE = "Y nulcline"

X_NUL_COLOR = 'blue'
Y_NUL_COLOR = 'orange'

X_NUL_LABEL = 'dx nulcline'
Y_NUL_LABEL = 'dy nulcline'

# critical points plot utilities
CRIT_POINT_LABEL = 'Critical point'
CRIT_POINT_COLOR = '#8C7073'
CRIT_POINT_STABLE_MARKER = 'o'
CRIT_POINT_SADDLE_MARKER = '$\diamondsuit$'
CRIT_POINT_UNSTABLE_SPIRAL_MARKER = '$\circlearrowright$'
CRIT_POINT_NON_ISOLATED_MARKER = '+'

# quiver plot utilities
ARROWS_LABEL = "system evolution"
ARROWS_COLOR = "red"


def plot_critical_points(
        graphic: plt,
        points: list,
        markers: list,
        color: str,
        label: str):
    '''
    This method will plot in a already
    created graphic all the critical points.

    :param graphic: The `plt` module with the
    started graphic in which we will be plotting
    our points.
    :param points: List of tuples representing
    the 2D coordinates of the points.
    :param markers: A list of strings indicating
    which markers correspond to each point.
    :param color: A string representing the color
    of all the points.
    :param label: The generic label for this type
    of plot, for instance: "Critical Point". Here
    we will add the point coordinates. In other words
    we will add to "Critical Point" the coordinates:
    "Critical Point (1, -3)".

    :return plt: The graphic with the plots.
    '''
    for point, marker in zip(points, markers):
        graphic.plot(
            point[0], point[1], color=color, marker=marker,
            label="{} ({:.2f},{:.2f})".format(
                label, point[0], point[1])
        )

    return graphic


def add_plot_properties(
        graphic: plt,
        x_lim: tuple = None,
        y_lim: tuple = None,
        x_label: str = None,
        y_label: str = None,
        title: str = None):
    '''
    This method will add a series of properties
    to a plot such as `x_lim`, `x_label`, ... Only
    if they are specified. 

    :param graphic: The `plt` module with the
    started graphic in which we will be plotting
    our points.
    :param x_lim: Limits on OX axis, defaults to None
    :param y_lim: Limits on OY axis, defaults to None
    :param x_label: OX label, defaults to None
    :param y_label: OY label, defaults to None
    :param title: Graphic's title, defaults to None
    '''
    graphic.xlim(x_lim)
    graphic.ylim(y_lim)
    graphic.xlabel(x_label)
    graphic.ylabel(y_label)
    graphic.title(title)

    return graphic


def plot_phase_space(
        graphic: plt,
        x_lim: tuple,
        y_lim: tuple,
        x_label: str,
        y_label: str,
        title: str,
        ox: np.array,
        X_grid: np.array,
        Y_grid: np.array,
        dx_nul: np.array,
        dx_nul_color: str,
        dx_nul_label: str,
        dy_nul: np.array,
        dy_nul_color: str,
        dy_nul_label: str,
        dx: np.array,
        x_factor: int,
        dy: np.array,
        y_factor: int,
        arrows_color: str,
        arrows_label: str,
        points: list = None,
        points_markers: list = None,
        points_color: str = None,
        points_label: str = None):
    '''
    This method will plot the phase
    space of a dynamic system.

    :param graphic: The `plt` module with the
    started graphic in which we will be plotting
    our points.
    :param x_lim: Limits on OX axis.
    :param y_lim: Limits on OY axis.
    :param x_label: OX label.
    :param y_label: OY label.
    :param title: Graphic's title.
    :param ox: X-axis.
    :param X_grid: X-meshgrid for the arrows.
    :param Y_grid: Y-meshgrid for the arrows.
    :param dx_nul: dx/dt nulcline.
    :param dx_nul_color: dx/dt nulcline color.
    :param dx_nul_label: dx/dt nulcline label.
    :param dy_nul: dy/dt nulcline.
    :param dy_nul_color: dy/dt nulcline color.
    :param dy_nul_label: dy/dt nulcline label.
    :param dx: Directions of the dx/dt arrows.
    :param x_factor: Increase factor of "dx".
    :param dy: Directions of the dy/dt arrows.
    :param y_factor: Increase factor of "dy".
    :param arrows_color: Arrows color.
    :param arrows_label: Arrows label.
    :param points: List of tuples representing
    the 2D coordinates of the points, defaults to None.
    :param points_markers: A list of strings indicating
    which markers correspond to each point,
    defaults to None.
    :param points_color: A string representing the color
    of all the points, defaults to None.
    :param points_label: The generic label for this type
    of plot, for instance: "Critical Point". Here
    we will add the point coordinates. In other words
    we will add to "Critical Point" the coordinates:
    "Critical Point (1, -3)", defaults to None.
    '''
    graphic.plot(
        ox, dx_nul, color=dx_nul_color,
        label=dx_nul_label)
    graphic.plot(
        ox, dy_nul, color=dy_nul_color,
        label=dy_nul_label)

    graphic.quiver(
        X_grid, Y_grid, x_factor*dx, y_factor*dy,
        color=arrows_color, label=arrows_label)

    graphic = plot_critical_points(
        graphic, points, points_markers,
        points_color, points_label)

    graphic = add_plot_properties(
        graphic, x_label=x_label,
        y_label=y_label,
        title=title,
        y_lim=y_lim,
        x_lim=x_lim)

    graphic.legend()
    return graphic


def plot_bifurcation_diagram(
        graphic: plt,
        x_curves: list,
        y_curves: list,
        curves_colors: list,
        curves_labels: list,
        critical_points: list,
        critical_points_markers: list,
        critical_points_label: str,
        linestyles: list):
    '''
    This method will print the bifurcation
    diagram.

    :param x_curves: List of curves ordered with
    the same indexes as `y_curves`. It will be
    a list of np.array's with all the trajectories.
    :param y_curves: List of curves ordered with
    the same indexes as `x_curves`. It will be
    a list of np.array's with all the trajectories.
    :param curves_colors: List of strings with the
    desired color per curve.
    :param curves_labels: List of strings with the
    desired label per curve.
    :param critical_points: All the critical points
    coordinates.
    which markers correspond to each point.
    :param critical_points_label: The generic label
    for this type of plot, for instance: "Critical Point".
    Here we will add the point coordinates. In other words
    we will add to "Critical Point" the coordinates:
    "Critical Point (1, -3)".
    :param linestyles: List with the linestyles per
    trajectory.

    :return graphic: Plot with the bifurcation diagram.
    '''
    for x_curve, y_curve, curve_color, curve_label, linestyle\
            in zip(x_curves, y_curves, curves_colors,
                   curves_labels, linestyles):
        graphic.plot(x_curve, y_curve,
                     color=curve_color,
                     label=curve_label,
                     linestyle=linestyle)

    for critical_point, critical_point_marker in zip(
            critical_points, critical_points_markers):
        graphic.plot(
            critical_point[0], critical_point[1],
            marker=critical_point_marker,
            label="{} ({:.2f},{:.2f})".format(
                critical_points_label,
                critical_point[0],
                critical_point[1])
        )

    return graphic
