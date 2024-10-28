# WEIS

[![Coverage Status](https://coveralls.io/repos/github/WISDEM/WEIS/badge.svg?branch=develop)](https://coveralls.io/github/WISDEM/WEIS?branch=develop)
[![Actions Status](https://github.com/WISDEM/WEIS/workflows/CI_WEIS/badge.svg?branch=develop)](https://github.com/WISDEM/WEIS/actions)
[![Documentation Status](https://readthedocs.org/projects/weis/badge/?version=develop)](https://weis.readthedocs.io/en/develop/?badge=develop)
[![DOI](https://zenodo.org/badge/289320573.svg)](https://zenodo.org/badge/latestdoi/289320573)

# BYU note

This is BYU's modified version of WEIS. The purpose is to be able to use a fatigue constraint in WISDEM optimization, where the damage is obtained from OpenFAST simulations. The coupling is loose, and can be iterated. Damage equavalent loads are also computed, and can be used as an input to MACH optimmization (that uses ADflow and TACS).

See dedicated examples in `./example_BYU`. *Case7* enables the computation of the damage equivalent loads, optionally runs an optimization with WISDEM, and optionally iterates over the previous 2 steps. *Case8* presents a method to extrapolate loads from low-fidelity to high-fidelity (ADflow-compatible) loadings. *Case9* includes various postprocessing and comparison scripts. See `./example_BYU/README.md` for more details.

# Original Readme

WEIS, Wind Energy with Integrated Servo-control, performs multifidelity co-design of wind turbines. WEIS is a framework that combines multiple NREL-developed tools to enable design optimization of floating offshore wind turbines.

Author: [NREL WISDEM & OpenFAST & Control Teams](mailto:systems.engineering@nrel.gov)

## Version

This software is a version 0.0.1.

## Documentation

See local documentation in the `docs`-directory or access the online version at <https://weis.readthedocs.io/en/latest/>

## Packages

WEIS integrates in a unique workflow four models:
* [WISDEM](https://github.com/WISDEM/WISDEM) is a set of models for assessing overall wind plant cost of energy (COE).
* [OpenFAST](https://github.com/OpenFAST/openfast) is the community model for wind turbine simulation to be developed and used by research laboratories, academia, and industry.
* [TurbSim](https://www.nrel.gov/docs/fy09osti/46198.pdf) is a stochastic, full-field, turbulent-wind simulator.
* [ROSCO](https://github.com/NREL/ROSCO) provides an open, modular and fully adaptable baseline wind turbine controller to the scientific community.

In addition, three external libraries are added:
* [pCrunch](https://github.com/NREL/pCrunch) is a collection of tools to ease the process of parsing large amounts of OpenFAST output data and conduct loads analysis.
* [pyOptSparse](https://github.com/mdolab/pyoptsparse) is a framework for formulating and efficiently solving nonlinear constrained optimization problems.

Software Model Versions:
Software        |       Version
---             |       ---
OpenFAST        |       3.2.1
ROSCO           |       2.6.0

The core WEIS modules are:
 * _aeroelasticse_ is a wrapper to call [OpenFAST](https://github.com/OpenFAST/openfast)
 * _control_ contains the routines calling the [ROSCO_Toolbox](https://github.com/NREL/ROSCO_toolbox) and the routines supporting distributed aerodynamic control devices, such trailing edge flaps
 * _gluecode_ contains the scripts glueing together all models and libraries
 * _multifidelity_ contains the codes to run multifidelity design optimizations
 * _optimization_drivers_ contains various optimization drivers
 * _schema_ contains the YAML files and corresponding schemas representing the input files to WEIS

## Installation

On laptop and personal computers, installation with [Anaconda](https://www.anaconda.com) is the recommended approach because of the ability to create self-contained environments suitable for testing and analysis.  WEIS requires [Anaconda 64-bit](https://www.anaconda.com/distribution/). WEIS is currently supported on Linux, MAC and Windows Sub-system for Linux (WSL). Installing WEIS on native Windows is not supported.

The installation instructions below use the environment name, "weis-env," but any name is acceptable. For those working behind company firewalls, you may have to change the conda authentication with `conda config --set ssl_verify no`.  Proxy servers can also be set with `conda config --set proxy_servers.http http://id:pw@address:port` and `conda config --set proxy_servers.https https://id:pw@address:port`.

**NOTE ON THE USE OF CONDA IN 2024** We start having big troubles because packages are too recent. I should have noted down the version of everything when it was working, but I didnt.
As a result:
- there is mostly a limitation on the numpy version for compatibility with WEIS (due to `distutil` being used). Python 3.12 does not even support it anymore so it will just not find it. We must employ Python<3.12 and most likely a numpy version 1.24 or 1.26 at most. But still, it might complain because of incompatibility with setuptools (<65?). The thing is: specigying the python version in the env file makes it unsolvable (unfinished after 12 hours...).
- alternately, we could you an old conda that relies on an older python... problem is: it will still try to suck a too recent python. 
Other notes:
- APPARENTLY, using python version in `env create` does not work anymore. Can still do in `conda create` but then you can't specify a file. 
- With Conda23.9, it's impossible to solve the environment with python3.9...
- Ways to create the env are:
        conda env create --name weis-env -f ./environment_byu.yml python=3.9
        ##-OR- for conda 23
        conda env create -y --name weis-env-1.1 -f ./environment_byu.yml
        ##-OR- for old conda
        <!-- #conda create --name weis-env-TMP2 --force --file ./environment_nic5.yml python=3.9 -->
        conda create --name weis-env-TMP2 -y python=3.9
        activate weis-env-TMP2
        conda install -y --file ./environment_nic5.yml

**INSTRUCTIONS Specific to nic5:**
```
conda config --add channels conda-forge
conda create --name weis-env-TMP2 -y python=3.9
conda activate weis-env-TMP2
conda install -y --file ./environment_nic5.yml
conda uninstall -y wisdem
#pip uninstall wisdem
pip install dearpygui marmot-agents
conda install -y petsc4py mpi4py ipopt pyoptsparse==2.10.2
python setup.py develop
```

1.  On the DOE HPC system eagle, make sure to start from a clean setup and type

        module purge
        module load conda        

2.  Setup and activate the Anaconda environment from a prompt (WSL terminal on Windows or Terminal.app on Mac)

        conda env create --name weis-env -f ./environment_byu.yml python=3.9
        conda activate weis-env                          # (if this does not work, try source activate weis-env)
        sudo apt update                                  # (WSL only, assuming Ubuntu)

3.  Use conda to add platform specific dependencies.

        conda config --add channels conda-forge
        conda uninstall wisdem                                               #there is probably still a package depending on wisdem in the environment file?
        pip uninstall wisdem           
        conda install -y petsc4py mpi4py                                     # (Mac / Linux only)   
        conda install -y compilers                                           # (Mac only)   
        sudo apt install gcc g++ gfortran libblas-dev liblapack-dev  -y      # (WSL only, assuming Ubuntu)
        
        conda uninstall pyhams
        pip uninstall pyHAMS
        conda install -y ipopt pyoptsparse==2.10.2

        <!-- conda install -y cmake cython control dill git jsonschema make matplotlib-base numpy==1.22 openmdao==3.16 openpyxl pandas pip pyoptsparse pytest python-benedict pyyaml ruamel_yaml scipy setuptools simpy slycot smt sortedcontainers swig
        pip install marmot-agents jsonmerge fatpack
        conda install -y pyhams statsmodels                              # (BYU specific?)  -->

        <!-- openmdao 3.16 requires numpy <=1.24 -->

**CAUTION** the current install gets all the packages from `raw.githubusercontent.com/WISDEM/WEIS`, including subpackages that are shipped in weis... and the shipped versions seem to supersede the local devel versions. So modifying the local code does not affect execution. To work that around, do NOT specify them in the environment file (or uninstall the packages from the conda env and pip).

1. Clone the repository and install the software

        git clone https://github.com/WISDEM/WEIS.git
        cd WEIS
        git checkout branch_name                         # (Only if you want to switch branches, say "develop")
        python setup.py develop                          # (The common "pip install -e ." will not work here)

2. Instructions specific for DOE HPC system Eagle.  Before executing the setup script, do:

        module load comp-intel intel-mpi mkl
        module unload gcc
        python setup.py develop

**NOTE:** To use WEIS again after installation is complete, you will always need to activate the conda environment first with `conda activate weis-env` (or `source activate weis-env`). On Eagle, make sure to reload the necessary modules

## Developer guide

If you plan to contribute code to WEIS, please first consult the [developer guide](https://weis.readthedocs.io/en/latest/how_to_contribute_code.html).

## Feedback

For software issues please use <https://github.com/WISDEM/WEIS/issues>.  
