
'''
This module purpose will be containing all the operations related
with the visualizations of images containing time series of a series
of neurons, ...
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import imageio
import datetime
import os


def plot_grid_cells(
        grid_walk: np.array,
        cells: list,
        linestyles: list):
    '''
    This function will print a time series per
    each neuron specified via arguments.

    :param grid_walk: The history of our neurons
    grid.
    :param cells: Neurons indexes that should be
    passed as a list of bi-dimensional tuples.
    :param linestyles: A list of linestyles to
    apply to each neuron in order to distinguish it
    from the rest.

    Example:
    ```
    grid_first_walk = random_walk(
        grid_first_sim_ini_vals, STEPS)

    plot_grid_cells(
        grid_first_walk,
        [(49, 2), (49, 3), (41, 0)],
        ['-', '--', '-.'])
    ```

    Output:
    ```
    <A time series of neurons>
    ```
    '''
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 1, 1)

    for cell, linestyle in zip(cells, linestyles):
        ax.plot(grid_walk[:, cell[0], cell[1]],
                linestyle=linestyle,
                label="# {}".format(cell))

    ax.set_xlabel("Time")
    ax.set_ylabel("Membrane Potential")
    ax.legend(loc="upper left")
    plt.show()


def generate_heatmap_t(
        grid_walk: np.array,
        instant: int,
        threshold: float,
        display=True):
    '''
    This method will print out a heatmap for a
    specified grid in a concrete instant.

    :param grid_walk: This will be a 3D array where
    the first dimension represents the instant (t),
    the second one the height of the grid, and the
    third one the width of the grid.
    :param instant: This will be the instant "t" to
    represent.
    :param threshold: This parameter will indicate
    from which point we will be representing a spike (red)
    or subthreshold activity (blue).
    :param display: This parameter will print in the
    notebook the heatmap or not, defaults to True

    Example:
    ```
    grid_walk = random_walk(matrix, STEPS)
    generate_heatmap_t(grid_walk, 100)
    ```

    Output:
    ```
    <The Heatmap display>
    ```

    :return plt: This method will return the graphic
    itself.
    '''
    vmin = np.min(grid_walk)
    vmax = np.max(grid_walk)

    # Definir colores
    colors_red = plt.cm.Reds(
        np.linspace(
            1, 0.3, int(threshold * 256 / vmax)))
    colors_blue = plt.cm.Blues(
        np.linspace(
            0.3, 1, int((vmax - threshold) * 256 / vmax)))
    colors = np.vstack((colors_red, colors_blue))

    cmap = ListedColormap(colors)

    # Create the heat map using the random walk
    # for the given time point
    heatmap = plt.imshow(
        grid_walk[instant],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    plt.axis('off')

    if display:
        # Add the colorbar
        plt.colorbar()

        # Add a title and remove axes
        plt.title('Instant heat map t={}'
                  .format(instant))

        # Display the heat map
        plt.show()
    else:
        return plt


def generate_gif_sequence(
        grid_walk: np.array,
        threshold: float,
        name: str,
        path: str = '',
        add_timestamp: bool = True):
    '''
    This method will use the `generate_heatmap_t`
    function to generate a gif over al the time
    steps specified by the `grid_walk` first dimension.

    The method will save the gif in: `<path>/`.

    :param grid_walk: This will be a 3D array where
    the first dimension represents the instant (t),
    the second one the height of the grid, and the
    third one the width of the grid.
    :param threshold: This parameter will indicate
    from which point we will be representing a spike (red)
    or subthreshold activity (blue).
    :param name: Name of the gif to be generated.
    :param path: Path where the gif will be stored,
    defaults to ''
    :param add_timestamp: This parameter will add to the
    gif name a timestamp, defaults to True


    Example:
    ```
    grid_walk = random_walk(grid, STEPS)
    generate_gif_sequence(
        grid_walk, L,
        'grid_walk', RESULTS_DIR,
        add_timestamp=False)
    ```

    Output:
    ```
    <Nothing, it will store the gif in the path specified>
    ```
    '''
    # Create an empty list to store the images
    images = []

    # Iterate through every instant in time
    for i in tqdm(range(grid_walk.shape[0])):
        # Create a heat map using the data matrix for this instant
        plt = generate_heatmap_t(
            grid_walk=grid_walk, instant=i, threshold=threshold, display=False)
        # Save the figure as an image and add the image to the list
        plt.savefig('temp.png', bbox_inches='tight', pad_inches=0)
        images.append(imageio.imread('temp.png'))

        # Clean the figure for the next iteration
        plt.clf()

    # Save the list of images as a GIF file
    if add_timestamp:
        name += '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
    name += '.gif'

    file_path = os.path.join(path, name)
    imageio.mimsave(file_path, images)
