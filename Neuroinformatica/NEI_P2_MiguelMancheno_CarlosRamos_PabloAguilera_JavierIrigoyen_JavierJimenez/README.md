# nei_p2
- [nei\_p2](#nei_p2)
  - [Authors](#authors)
  - [Description](#description)
  - [Recommendations](#recommendations)
  - [Instructions](#instructions)
  - [Execution environment](#execution-environment)

## Authors
- Carlos Ramos Mateos
- Miguel Ángel Mancheño Yustres
- Pablo Aguilera Onieva
- Javier Irigoyen Muñoz
- Javier Jiménez Rodríguez

## Description
In this file we will describe the hierarchical structure of the project, the minimum requisites to execute the project, the execution environment and a series of recommendations to evaluate it.

## Recommendations
We encourage the evaluator to evaluate the practice in the following order:
1. `HR_2D.ipynb` (or its `*.html` equivalent) notebook which defines the Hindmarsh-Rose 2D dynamics with a series of explainations of the bifurcation diagram, the critical points, and a final conclusion that could be applied to the 3D system (with the extra $z$ equation).
2. `HR_3D.ipynb` (or its `*.html` equivalent) notebook which defines the Hindmarsh-Rose 3D dynamics with a series of explainations of the system evolution along time, the $z$ influence, and how we managed to replicate the HR-2D mechanics.

## Instructions
In this project we provide the following structure:
```
.
├── README.md
├── reports
│   ├── HR_2D.html
│   └── HR_3D.html
└── workspace
    ├── HR_2D.ipynb
    ├── HR_3D.ipynb
    └── lib
        ├── dyn_sys_utils.py
        ├── hindmarsh_rose.py
        ├── hr_2d.py
        ├── plot_utils.py
        └── Utils.py
```

Now we will explain each file and folder:
- `reports/`, this folder contains a HTML version of the project to avoid executing them to visualize the results.
  - `HR_*D.html`, notebook with the conclusions of the study.
- `workspace/`, this folder contains the `*.ipynb` notebooks with our explainations and results (some have to be executed in real time to obtain the results, if you want to avoid this, check the results in the `reports/` folder).
  - `HR_*D.ipynb`, jupyter notebook with the conclusions and code to replicate the results.
  - `lib/`, folder with all the modules & functions used in the jupyter notebooks.
    - `dyn_sys_utils.py`, this module objective will be obtaining characteristics of a specific dynamical system in order to explain it.
    - `hindmarsh_rose.py`, this module defines the Hindmarsh-Rose system dynamics in both 2D & 3D.
    - `hr_2d.py`, in this module we will define all the methods to simulate a Hindmarsh-Rose 2D system.
    - `plot_utils.py`, this module will add some methods responsible of the graphics visualization.
    - `Utils.py`, this module will define a series of general extra functions useful for the 2D and 3D system.

## Execution environment
To execute the notebooks it should be enough by installing the following Python environment in Anaconda:
```
anyio==3.6.2
argon2-cffi==21.3.0
argon2-cffi-bindings==21.2.0
arrow==1.2.3
asttokens==2.2.1
attrs==23.1.0
autopep8==2.0.2
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
importlib-metadata==6.6.0
importlib-resources==5.12.0
ipykernel==6.22.0
ipympl==0.9.3
ipython==8.12.2
ipython-genutils==0.2.0
ipywidgets==8.0.6
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
jupyterlab-widgets==3.0.7
kiwisolver==1.4.4
MarkupSafe==2.1.2
matplotlib==3.7.1
matplotlib-inline==0.1.6
mccabe==0.7.0
mistune==2.0.5
nbclassic==1.0.0
nbclient==0.7.4
nbconvert==7.3.1
nbformat==5.8.0
nest-asyncio==1.5.6
notebook==6.5.4
notebook_shim==0.2.3
numpy==1.24.3
packaging==23.1
pandocfilters==1.5.0
parso==0.8.3
pexpect==4.8.0
pickleshare==0.7.5
Pillow==9.5.0
pkgutil_resolve_name==1.3.10
platformdirs==3.5.0
prometheus-client==0.16.0
prompt-toolkit==3.0.38
psutil==5.9.5
ptyprocess==0.7.0
pure-eval==0.2.2
pycodestyle==2.10.0
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
scipy==1.10.1
Send2Trash==1.8.2
six==1.16.0
sniffio==1.3.0
soupsieve==2.4.1
stack-data==0.6.2
terminado==0.17.1
tinycss2==1.2.1
tomli==2.0.1
tornado==6.3.1
tqdm==4.65.0
traitlets==5.9.0
typing_extensions==4.5.0
uri-template==1.2.0
wcwidth==0.2.6
webcolors==1.13
webencodings==0.5.1
websocket-client==1.5.1
widgetsnbextension==4.0.7
zipp==3.15.0
```
