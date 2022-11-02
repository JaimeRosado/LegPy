# Author: Fernando Arqueros

import LegPy as lpy
import numpy as np
import matplotlib.pyplot as plt
import random as rand
import math

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources
from . import beam_spectra as bs

def Spectrum(name='mono', E=None, E_w=None, E_min=None, E_max=None, E_mean=None, E_sigma=None, E_ch=None, file=None):
    
    if name == 'mono':
        if E is None:
            raise ValueError('Please, input E.')
        spectrum = Mono(E)  
    
    elif name == 'multi_mono':
        if E_w is None:
            raise ValueError('Please, input E_w.')
        spectrum = Multi_mono(E_w)

    elif name == 'flat':
        if E_min is None or E_max is None:
            raise ValueError('Please, input E_min and E_max.')
        spectrum = Flat(E_min, E_max)

    elif name == 'gaussian':
        if E_mean is None or E_sigma is None:
            raise ValueError('Please, input E_mean and E_sigma.')
        spectrum = Gaussian(E_mean, E_sigma)

    elif name == 'exponential':
        if E_min is None or E_max is None or E_ch is None:
            raise ValueError('Please, input E_min, E_max and E_ch.')
        spectrum = Exponential(E_min, E_max, E_ch)

    elif name == 'reciprocal': # ~ 1/E
        if E_min is None or E_max is None:
            raise ValueError('Please, input E_min and E_max.')
        spectrum = Reciprocal(E_min, E_max)

    elif name == 'from_file': # provided by a txt file (E_i , w_i) w_i = weight of energy E_i.
        if file is None:
            raise ValueError('Please, input file.')
        try:
            spect_arr = np.loadtxt(file)
        except:
            try:
                spect_arr = np.loadtxt(pkg_resources.open_text(bs, file))
            except:
                raise ValueError('File not found.')
        spectrum = From_file(spect_arr)

    else: 
        raise ValueError('Spectrum type not available')
        
    return spectrum

class Mono:
    def __init__(self, E):
        self.name = "mono"
        self.E = E
        self.E_max = E
    def in_energy(self):
        return self.E
    def plot(self, n_part = 100000, n_bin = 50):
        return Plot_beam_spectrum(self, n_part, n_bin)
    
class Gaussian(Mono):
    def __init__(self, E_mean, E_sigma):
        self.name = "gaussian"
        self.E_mean = E_mean
        self.E_sigma = E_sigma
        self.E_max = E_mean + E_sigma
    def in_energy(self):
        while True:
            E = rand.gauss(0., self.E_sigma)
            if abs(E) <= self.E_mean: 
                return E + self.E_mean

class Multi_mono(Mono):
    def __init__(self, E_w):
        self.name = "multi_mono"
        self.energies, self.weights = E_w[:,0], E_w[:,1].cumsum()
        self.weights /= self.weights[-1]
        self.E_max = self.energies.max()
    def in_energy(self):
        rn = rand.random()
        if rn==1:
            return self.energies[-1]
        for E, w in zip(self.energies, self.weights):
            if rn < w:
                return E

class Flat(Mono):
    def __init__(self, E_min, E_max):
        self.name = "flat"
        self.E_min = E_min
        self.E_max = E_max
    def in_energy(self):
        return rand.random() * (self.E_max - self.E_min) + self.E_min

class Exponential(Mono):
    def __init__(self, E_min, E_max, E_ch):
        self.name = "exponential"
        self.E_min = E_min
        self.E_max = E_max
        self.E_ch = E_ch
        self.p_min = math.exp(- E_min / E_ch)
        self.p_min_max = self.p_min - math.exp(- E_max / E_ch)
    def in_energy(self):
        return - self.E_ch * math.log(self.p_min - self.p_min_max * rand.random())

class Reciprocal(Mono):
    def __init__(self, E_min, E_max):
        self.name = "reciprocal"
        self.l_min = math.log(E_min)
        self.l_max = math.log(E_max)
        self.E_max = E_max
    def in_energy(self):
        return math.exp(self.l_min + (self.l_max - self.l_min)*rand.random()) 

class From_file(Mono):
    def __init__(self, spect_arr):
        self.name = "from_file"
        self.spect_arr = spect_arr
        self.Imax= np.amax(spect_arr, axis = 0)[1] # maximum intensity
        self.E_max = np.amax(spect_arr, axis = 0)[0] + (spect_arr[1,0]-spect_arr[0,0]) / 2. ##FAM add half bin
        self.E_min = np.amin(spect_arr, axis = 0)[0] - (spect_arr[1,0]-spect_arr[0,0]) / 2. ##FAM substract half bin

    def in_energy(self):
        while True:        
            E = self.E_min + (self.E_max - self.E_min) * rand.random() # random E
            Int = self.Imax * rand.random() # random ordinate
            I = np.interp(E, self.spect_arr[:,0], self.spect_arr[:,1]) # I(E) from file spectrum
            if I > Int:
                return E

def Plot_beam_spectrum(spectrum, n_part, n_bin):
    ener_arr = [spectrum.in_energy() for n in range(n_part)]

    #plt.hist(ener_arr, n_bin, density = True, color = 'b')
    plt.hist(ener_arr, bins=np.logspace(start=np.log10(0.001), stop=np.log10(20.), num=n_bin), density = True)
    
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Probability' + ' (MeV$^{-1}$)')
    plt.title('Energy spectrum of incident gamma-ray beam')
    plt.xlim(0.001, 20.)
    plt.grid(True, which = 'both')
    # plt.legend()
    plt.xscale('log')
    #plt.gca().set_xscale("log")
    plt.show()