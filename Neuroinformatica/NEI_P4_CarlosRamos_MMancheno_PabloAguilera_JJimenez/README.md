# Tabla de contenidos
- [Tabla de contenidos](#tabla-de-contenidos)
- [NEI - P4](#nei---p4)
  - [Autores](#autores)
  - [Distribución de tareas](#distribución-de-tareas)
    - [Tareas de Miguel Ángel](#tareas-de-miguel-ángel)
    - [Tareas de Carlos Ramos Mateos](#tareas-de-carlos-ramos-mateos)
    - [Tareas de Pablo Aguilera Onieva](#tareas-de-pablo-aguilera-onieva)
    - [Tareas de Javier Jiménez Rodríguez](#tareas-de-javier-jiménez-rodríguez)
  - [Resultados](#resultados)
  - [Instrucciones](#instrucciones)
  - [Jerarquía](#jerarquía)
    - [Imágenes y GIFs](#imágenes-y-gifs)
    - [Código](#código)

# NEI - P4
## Autores
- Miguel Ángel Mancheño Yustres.
- Carlos Ramos Mateos.
- Pablo Aguilera Onieva.
- Javier Jiménez Rodríguez.

## Distribución de tareas
Para distribuir las tareas entre los cuatro miembros se completaron las tareas inicialmente definidas por los integrantes de un [tablero Trello](https://trello.com/invite/b/phUj23Uv/ATTI37fd2c7d11261888daf754366c88afa571437B34/nei-practice-4).

No obstante definiremos las tareas de las que se ocupó cada miembro en las siguientes subsecciones.

### Tareas de Miguel Ángel
Miguel Ángel se ocupó de comenzar y/o completar las siguientes tareas:
- Todo lo relacionado con la visualización de mapas de calor.
- Una parte de la integración entre el caminante aleatorio y la evolución de la rejilla de neuronas.
  - En gran parte ayudando a Carlos Ramos.
- Todo lo relacionado con el guardado y carga del progreso de la rejilla para no ejecutarlo varias veces.
- El encapsulado de las simulaciones en GIFs.
- La lectura de los artículos:
  - Biophysical Properties of Subthreshold Resonance Oscillations and Subthreshold Membrane Oscillations in Neurons. (15 páginas)
  - Oscillation and noise determine signal transduction in shark multimodal sensory cells. (4 páginas)
- Aportar ideas en la reunión final en la que discutir y agrupar todas las conclusiones finales.
- Comunicar a Francisco de Borja las dudas del equipo.

### Tareas de Carlos Ramos Mateos
Carlos Ramos se ocupó de comenzar y/o completar las siguientes tareas:
- Gran parte de la integración entre el caminante aleatorio y la evolución de la rejilla de neuronas.
  - Sobre todo en lo que a calcular el siguiente paso de una neurona en la simulación se refiere teniendo en cuenta las diferentes mecánicas de la neurona (subumbral, spiking, ...).
- La lectura de los artículos:
  - Electrotonically Mediated Oscillatory Patterns in Neuronal Ensembles An In Vitro Voltage-Dependent Dye-Imaging Study in the Inferior Olive. (12 páginas)
  - Subthreshold Oscillations of the Membrane Potential A Functional Synchronizing and Timing Device. (6 páginas)
- Aportar ideas en la reunión final en la que discutir y agrupar todas las conclusiones finales.

### Tareas de Pablo Aguilera Onieva
Pablo Aguilera se ocupó de comenzar y/o completar las siguientes tareas:
- Aportar ideas en la reunión final en la que discutir y agrupar todas las conclusiones finales.
- La lectura del artículo:
  - Transient dynamics and rhythm coordination of inferior olive spatio-temporal patterns. (18 páginas)
- Modificar el método de la generación del mapa de calor para añadir los gradientes adecuados y la frontera entre spike (colores azules) y mecánica subumbral (colores rojos).
  - Con esto nos referimos al método original de Miguel Á. Mancheño.
- Rellenado del librillo final con todos los resultados y simulaciones.

### Tareas de Javier Jiménez Rodríguez
Javier Jiménez se ocupó de comenzar y/o completar las siguientes tareas:
- Aportar ideas en la reunión final en la que discutir y agrupar todas las conclusiones finales.
- Distribuir los artículos en las etapas iniciales del proyecto.
- La creación de módulos, documentación y estructura del proyecto.
- Crear diccionarios, métodos de inicialización de la rejilla y enumeraciones (`ParameterEnum.py`) para estandarizar el trabajo de todos los integrantes y "hablar en un lenguaje común".
- La lectura de los artículos:
  - Temporal Neuronal Oscillations can Produce Spatial Phase Codes. (11 páginas)
  - Stochastic Networks with Subthreshold Oscillations and Spiking Activity. (7 páginas)
- Modificar el método de la generación del mapa de calor para añadir los gradientes adecuados y la frontera entre spike (colores azules) y mecánica subumbral (colores rojos).
  - Con esto nos referimos al método original de Miguel Á. Mancheño.
- Preparación de una plantilla que, finalmente, se convirtió en `NEI_P4_Report.ipynb` con las rejillas de las diferentes simulaciones preinicializadas para permitir al resto de compañeros hacer "Plug&Play" de sus implementaciones.
- Preparación de la entrega final así como la jerarquía y el repositorio de GIFs del proyecto (las simulaciones).
- Rellenado del librillo final con todos los resultados y simulaciones.

## Resultados
Dado que los GIFs generados en la práctica son muy pesados (~ 50 MBs cada uno), facilitamos un enlace a nuestro repositorio con dichas simulaciones en OneDrive puesto que, cada GIF, tarda en ser generado ~ 2-3 minutos.

Para evitar dicha espera los GIFs pueden ser consultados en el siguiente enlace:
[OneDrive - NEI P4 Results](https://dauam-my.sharepoint.com/:f:/g/personal/javier_jimenez02_estudiante_uam_es/EtJcCLlJHz9IiRdOJpfx1SUB7j20NJs43sos0jFTM_UPFg?e=IU5Bxv)

De lo contrario, pueden ser generados ejecutando el librillo `P4_NEI.ipynb` al completo.

## Instrucciones
Para visualizar adecuadamente los GIFs en el librillo adjunto será necesario arrastrarlos a la carpeta `results/` del espacio de trabajo.

Este proyecto puede ser ejecutado con Python 3.8.10 y el siguiente entorno de trabajo utilizando Anaconda:
```
anyio==3.6.2
argon2-cffi==21.3.0
argon2-cffi-bindings==21.2.0
arrow==1.2.3
asttokens==2.2.1
attrs==23.1.0
autopep8 @ file:///opt/conda/conda-bld/autopep8_1650463822033/work
backcall==0.2.0
beautifulsoup4==4.12.2
bleach==6.0.0
cffi==1.15.1
comm==0.1.3
contourpy==1.0.7
cycler==0.11.0
debugpy==1.6.7
decorator==5.1.1
defusedxml==0.7.1
executing==1.2.0
fastjsonschema==2.16.3
flake8==6.0.0
fonttools==4.39.3
fqdn==1.5.1
idna==3.4
imageio==2.28.0
importlib-metadata==6.6.0
importlib-resources==5.12.0
ipykernel==6.22.0
ipython==8.12.0
ipython-genutils==0.2.0
isoduration==20.11.0
jedi==0.18.2
Jinja2==3.1.2
jsonpointer==2.3
jsonschema==4.17.3
jupyter-events==0.6.3
jupyter_client==8.2.0
jupyter_core==5.3.0
jupyter_server==2.5.0
jupyter_server_terminals==0.4.4
jupyterlab-pygments==0.2.2
kiwisolver==1.4.4
MarkupSafe==2.1.2
matplotlib==3.7.1
matplotlib-inline==0.1.6
mccabe==0.7.0
mistune==2.0.5
nbclassic==0.5.5
nbclient==0.7.3
nbconvert==7.3.1
nbformat==5.8.0
nest-asyncio==1.5.6
notebook==6.5.4
notebook_shim==0.2.2
numpy==1.24.3
packaging==23.1
pandocfilters==1.5.0
parso==0.8.3
pexpect==4.8.0
pickleshare==0.7.5
Pillow==9.5.0
pkgutil_resolve_name==1.3.10
platformdirs==3.2.0
prometheus-client==0.16.0
prompt-toolkit==3.0.38
psutil==5.9.5
ptyprocess==0.7.0
pure-eval==0.2.2
pycodestyle @ file:///croot/pycodestyle_1674267221883/work
pycparser==2.21
pyflakes==3.0.1
Pygments==2.15.1
pyparsing==3.0.9
pyrsistent==0.19.3
python-dateutil==2.8.2
python-json-logger==2.0.7
PyYAML==6.0
pyzmq==25.0.2
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
Send2Trash==1.8.0
six==1.16.0
sniffio==1.3.0
soupsieve==2.4.1
stack-data==0.6.2
terminado==0.17.1
tinycss2==1.2.1
toml @ file:///tmp/build/80754af9/toml_1616166611790/work
tornado==6.3.1
tqdm==4.65.0
traitlets==5.9.0
typing_extensions==4.5.0
uri-template==1.2.0
wcwidth==0.2.6
webcolors==1.13
webencodings==0.5.1
websocket-client==1.5.1
zipp==3.15.0
```

## Jerarquía
Para dejarlo todo como el espacio de trabajo original será necesario replicar la siguiente jerarquía:

    ```
    .
    ├── imgs
    │   ├── BurgessC2011 Cyclic information integration system.png
    │   ├── BurgessC2011 Information integration based on frequency variation.png
    │   ├── CastellanosN2003 - Isolated neuron behaviours.png
    │   ├── CastellanosN2003 - Spatio-temporal patterns displayed by the networks of 50x50 neurons.png
    │   ├── CastellanosN2003 - Subthreshold oscillation and spiking activity for three neurons in a population of 50x50 units.png
    │   ├── Place cell firing of a rat in a enclosure.png
    │   └── Trajectory of the animal in the enclosure while grid cells fire.png
    ├── NEI_P4_Report.ipynb
    ├── README.md
    ├── results
    │   ├── .gitkeep
    │   ├── grid_fifth_walk.gif
    │   ├── grid_first_walk.gif
    │   ├── grid_fourth_walk.gif
    │   ├── grid_second_walk.gif
    │   ├── grid_sixth_walk.gif
    │   └── grid_third_walk.gif
    └── utils
        ├── common.py
        ├── grid_utils.py
        ├── initialization_functions.py
        ├── initialization.py
        ├── ParameterEnum.py
        ├── plot_utils.py
        ├── rw_utils.py
        └── stimulation.py
    ```

Hemos de recordar que, de toda la jerarquía adjuntada anteriormente, los únicos ficheros no añadidos (por problemas de espacio en la entrega de Moodle) son los ficheros \*.gif (que pueden ser encontrados en el enlace mencionado en la sección [resultados](#resultados)).

### Imágenes y GIFs
En cuanto a imágenes y GIFs:
- `imgs/` contiene las imágenes utilizadas para defender/apoyar nuestras conclusiones. Estas imágenes han sido extraídas de nuestras simulaciones y de los siguientes artículos:

  - Stochastic Networks with Subthreshold Oscillations and Spiking Activity:
    ```
    @book_section{CastellanosN2003,
       author = {Nazareth P. Castellanos and Francisco B. Rodríguez and Pablo Varona},
       doi = {10.1007/3-540-44868-3_5},
       pages = {32-39},
       title = {Stochastic Networks with Subthreshold Oscillations and Spiking Activity},
       url = {http://link.springer.com/10.1007/3-540-44868-3_5},
       year = {2003},
    }
    ```

  - Temporal Neuronal Oscillations can Produce Spatial Phase Codes:
    ```
    @book_section{BurgessC2011,
        author = {Christopher Burgess and Nicolas W Schuck and Neil Burgess},
        doi = {10.1016/B978-0-12-385948-8.00005-0},
        journal = {Space, Time and Number in the Brain},
        pages = {59-69},
        publisher = {Elsevier},
        title = {Temporal Neuronal Oscillations can Produce Spatial Phase Codes},
        url = {https://linkinghub.elsevier.com/retrieve/pii/B9780123859488000050},
        year = {2011},
    }
    ```

- `results/` debería contener los GIFs con las simulaciones de los mapas de calor. Aunque, como ya hemos mencionado, por cuestiones de espacio no hemos podido adjuntarlos en la entrega, para más información revise la sección de [resultados](#resultados).


### Código
En lo que al código se refiere, hemos dividido el proyecto en los siguientes módulos (en los cuales se puede encontrar disponible la documentación de cada método):
- `common.py` es el módulo que implementa funciones de utilidad comúnes como transformar un diccionario en un `np.array` en función de la enumeración `ParameterEnum`.
- `grid_utils.py` contiene funciones de utilidad a la hora de comprobar diferentes celdas de la rejilla.
- `initialization_functions.py` define una serie de funciones para inicializar la rejilla con diferentes métodos como utilizando diccionarios, ...
- `initialization.py` contiene los métodos encargados de inicializar la rejilla con valores comúnes, puntuales, ...
- `ParameterEnum.py` define la enumeración común utilizada a lo largo de todos los módulos para identificar qué índices de la rejilla corresponden con cada valor ($a_i(t), a_i(t-1), ...$).
- `plot_utils.py` implementa una serie de funciones de utilidad para crear mapas de calor, representar series temporales, ...
- `rw_utils.py` contiene funciones que manipulan, modifican y guardan el proceso iterativo del avance de la rejilla utilizando un caminante aleatorio con barreras absorbentes.
- `stimulation.py` define un par de funciones para crear clústeres de estimulación, sumideros, ..., alrededor de una neurona marcada como "referencia" que tomaremos como tal para construir el clúster completo y distinguirlo del resto de la rejilla.