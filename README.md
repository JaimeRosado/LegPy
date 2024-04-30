# LegPy

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8414274.svg)](https://doi.org/10.5281/zenodo.8414274)

Low energy gamma-ray simulation with Python.

--------

## Introduction

LegPy is a Monte Carlo (MC) simulation package for the transportation of gamma rays and electrons through any material medium. The algorithm includes several approximations that accelerate the simulation while maintaining reasonably accurate results. Notably, pair production and Bremsstrahlung are ignored, which limits the applicability of the algorithm to low energies ($\lesssim 5$ MeV, depending on the medium). The package includes a library of media containing all the necessary data taken from NIST databases. Several simple geometries of the target object are supported (cylinder, orthohedron and sphere). The target object can be filled either with one single medium or contained two media splitted by an interface surface. Photons or electrons are produced as either parallel/divergent beams or from isotropic sources with energies following any spectral distributions.

The description of the MC algorithm and the validation of the various approximations are described in https://doi.org/10.1016/j.radmeas.2023.107029. In the latest version of the code, the generation of atomic $K_\alpha$ y $K_\beta$ X-rays after photoelectric absorption has been implemented, extending the validation of the algorithm beyond that shown in the above-mentioned paper.

## How to use

LegPy provides a user-friendly tool to perform simple simulations with just a few lines of code. It is designed to be used in Google Colab (https://colab.research.google.com/) or Jupyter (https://jupyter.org/) notebooks, which are interactive environments that improve user experience without requiring programming skills. Indeed, LegPy is used for educational purposes in both the Physics Degree and Master's Degree in Biomedical Physics at the Complutense University of Madrid.

The package includes an exhaustive [tutorial notebook](https://github.com/JaimeRosado/LegPy/blob/main/LegPy_tutorial.ipynb) that can be used in both frameworks. The tutorial includes a number of examples and procedures to visualize and analyze the results of the simulation. We recommend using this [tutorial notebook](https://github.com/JaimeRosado/LegPy/blob/main/LegPy_tutorial.ipynb) as a template.

### Colab

From a Colab notebook, clone this repository and add the directory where LegPy is located to the Python search path:
```
!git clone https://github.com/JaimeRosado/LegPy.git
import sys
sys.path.insert(0,'/content/LegPy')
```
Then import the package:
```
import LegPy as lpy
```

### Jupyter

Download this repository to your PC. Create a Jupyter notebook in the directory where the package is located (or use the [tutorial notebook](https://github.com/JaimeRosado/LegPy/blob/main/LegPy_tutorial.ipynb) included in the repository). Then import the package:
```
import LegPy as lpy
```
To use LegPy, Matplotlib, Pandas and Numpy need to be installed.
