import pickle
import numpy as np
from matplotlib import pyplot as plt

# Funciones para el guardado de DataFrames en disco usando Pickle


def save_obj(obj, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(obj, file)


def load_obj(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

# Función para el dibujado de señales


def plot_signal(potencial, primeros_spikes=None, puntos_hiperpol=None,
                title="Title not defined", x_label="", y_label=""):
    plt.figure(figsize=(15, 5), facecolor="w")
    with plt.style.context('classic'):
        # plt.plot(potencial, color="red")
        plt.plot(*zip(*potencial), color="red")
        if primeros_spikes is not None:
            plt.scatter(*zip(*primeros_spikes), color="blue")
        if puntos_hiperpol is not None:
            plt.scatter(*zip(*puntos_hiperpol), color="green")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.show()


# Función para el dibujado de la señal en 3D

def plot_signal_3d(x_array, y_array, z_array, title='Title not defined'):

    # ax = plt.axes(projection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xline = [x[1] for x in x_array]
    yline = [y[1] for y in y_array]
    zline = [z[1] for z in z_array]

    ax.set_title(title)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.plot3D(xline, yline, zline)

    plt.show()


# Función para el cálculo incremental de la media


def incremental_average(new_element, old_average=0, num_elements=1):
    return old_average + ((new_element - old_average) / num_elements)

# Función que devuelve una lista de desviaciones típicas
# medidas en una ventana de 5 elementos
# Devuelve un segundo valor que indica si la señal es estable
#  (si los últimos 5 valores son inferiores a e_regular)


def calc_desv_est(puntos_hiperpol, e_regular):
    periodos = []
    est_list = []
    for i in range(len(puntos_hiperpol)-1):
        periodo = puntos_hiperpol[i+1][0]-puntos_hiperpol[i][0]
        periodos.append(periodo)
    for i in range(len(periodos)-5):
        est_list.append(np.std(periodos[i:i+5]))

    estabilizada = all(estdv <= e_regular for estdv in est_list[-5:])

    return est_list, estabilizada

#Imprimir ecuaciones del sistema dinámico de dos dimensiones
def print_equations(i:int, z:int, x_sols:np.array, equations_sols:np.array):
    
    if z-i < 0:
        c = "- {}".format((z-i)*(-1))
    elif z-i == 0:
        c=''
    else: c= "+ {}".format(z-i)
  
    print("First equation: y_1 = lambda x: x*x*x - 2.6*x*x {}".format(c))
    print("\tSolutions:\n\t x: {}\n\t y: {}".format(x_sols, equations_sols[0]))
 
    print("\nSecond equation: y_2 = lambda x: 1 - 5*x*x")
    print("\tSolutions:\n\t x: {}\n\t y: {}".format(x_sols, equations_sols[1]))
 
    print("\nThey are both the same!")
