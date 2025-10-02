import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
import random
from .analysis_tools import ext_range


class esc_gammas:
    # Plot theta vs energy of escaped photons
    def __init__(self, E_max):
        self.E_max = E_max
        self.points = np.array([[0., 0.]])
        
    def add_count(self, output):
        hist_esc, E_ab, E, theta = output
        if hist_esc:
            self.points = np.append(self.points, [[theta, E]], axis = 0)
        
    def plot(self):
        points = self.points[1:] # delete inital [0,0]
        plt.figure()
        plt.scatter(points[:,0] / math.pi, points[:,1], marker = '.')
        plt.xlabel(r'Angle ('+'\u03C0'+'$\cdot$'+'rad)')
        plt.ylabel('Energy (MeV)')
        plt.xlim(0., 1.)
        plt.ylim(0., 1.05 * self.E_max)
        plt.title('Angle vs. energy for outgoing photons')
        plt.grid(True, which = 'both')


class hist:
    # Fill histogram of n in runtime
    def __init__(self, n, val_max, val_min=0.):
        self.n = n
        self.i_max = n - 1
        self.hist = np.zeros(n)
        self.val_max = val_max
        self.val_min = val_min
        self.delta = (val_max - val_min) / n
        self.left = 0.
        self.right = 0.

    def add_count(self, val, counts=1):
        if val<self.val_min:
            self.left += counts
        elif val>self.val_max:
            self.right +=counts
        else:
            val = val - self.val_min
            i = min(self.i_max, int(val / self.delta))
            self.hist[i] += counts


class e_hists:
    # Histograms of final z and maximum z of electrons
    # Histogram of theta angle for backscattered electrons (z<0)
    def __init__(self, n_z, n_ang, z_top, tot_n_part):
        self.n_z = n_z
        self.n_ang = n_ang
        self.z_top = z_top
        self.tot_n_part = tot_n_part
        self.delta_z = z_top / n_z
        self.delta_ang = math.pi / 2. / n_ang
        self.range_hist = hist(n_z, z_top) # final z
        self.trans_hist = hist(n_z, z_top) # maximum z
        self.back_hist = hist(n_ang, math.pi, math.pi/2.) # theta of backscattered electrons
        self.max_depth = 0.
        self.z_bin = np.arange(self.delta_z/2., self.z_top, self.delta_z)
        self.ang_bin = np.arange(math.pi/2. + self.delta_ang/2., math.pi, self.delta_ang)
        
    def add_count(self, output):
        e_in, E, z_max, position, theta = output
        z = position[2]
        self.range_hist.add_count(z)
        self.trans_hist.add_count(z_max)
        if z_max>self.max_depth:
            self.max_depth = z_max # Shown at the end of the simulation
        if not e_in and z<0.:
            self.back_hist.add_count(theta)
        
    def plot(self):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

        # Histogram of final z 
        range_hist = self.range_hist.hist
        range_coef = 1. - self.range_hist.hist.cumsum() / self.range_hist.hist.sum() # backscattered electrons excluded
        ax[0].bar(self.z_bin, range_hist, width = self.delta_z)
        ax[0].set_xlabel('Depth (cm)')
        ax[0].set_ylabel('Number of electrons')
        ax[0].set_title('Range of electrons')
        
        # Histogram of max z
        trans_hist = self.trans_hist.hist
        trans_coef = 1. - self.trans_hist.hist.cumsum() / self.trans_hist.hist.sum()  # backscattered electrons excluded
        print('Maximum depth (cm): ', round(self.max_depth, 3))
        ax[1].scatter(self.z_bin, trans_coef, s = 25)
        ax[1].set_xlabel('Depth (cm)')
        ax[1].set_ylabel('Fraction of electrons')
        ax[1].set_title('Transmission coefficient')
        ax[1].set_xlim(xmin = 0.)
        ax[1].set_ylim(ymin = 0.)

        # Histogram of theta for backscattered electrons
        back_hist = self.back_hist.hist
        back_hist_solid = back_hist / self.delta_ang / self.tot_n_part / (2. * math.pi * np.sin(self.ang_bin))
        tot_back = back_hist.sum()
        print('Fraction of backscattered electrons: ', round(tot_back/self.tot_n_part, 3))
        ax[2].bar(self.ang_bin / math.pi, back_hist, width = self.delta_ang / math.pi)
        ax[2].set_title('Angular spectrum of backscatered electrons')
        ax[2].set_xlabel(r'Angle ('+'\u03C0'+'$\cdot$'+'rad)')
        #ax[2].set_xlim(0., 1.)
        ax[2].set_ylabel('Number of electrons')

    def final_z(self):
        # Histogram of final z
        range_hist = self.range_hist.hist
        range_coef = 1. - self.range_hist.hist.cumsum() / self.range_hist.hist.sum() # backscattered electrons excluded
        range_df = np.column_stack((self.z_bin, range_hist, range_coef))
        range_df = pd.DataFrame(range_df, columns = ['z/cm', 'electrons', 'fraction'])
        return range_df

    def max_z(self):
        # Histogram of max z
        trans_hist = self.trans_hist.hist
        trans_coef = 1. - self.trans_hist.hist.cumsum() / self.trans_hist.hist.sum()  # backscattered electrons excluded
        trans_df = np.column_stack((self.z_bin, trans_hist, trans_coef))
        trans_df = pd.DataFrame(trans_df, columns = ['z/cm', 'electrons', 'fraction'])
        return trans_df

    def ext_range(self, definition="final"):
        if definition=="max":
            df = self.max_z()
        else:
            df = self.final_x()
        return ext_range(df)     

    def backscattering(self):
        # Histogram of theta for backscattered electrons
        back_hist = self.back_hist.hist
        back_hist_solid = back_hist / self.delta_ang / self.tot_n_part / (2. * math.pi * np.sin(self.ang_bin))
        tot_back = back_hist.sum()
        back_df = np.column_stack((self.ang_bin, back_hist, back_hist_solid))
        back_df = pd.DataFrame(back_df, columns = ['angle/rad', 'electrons', 'dn/dOmega'])
        return back_df

class gamma_hists:
    # Histogram of absorbed energy
    # Histograms of theta and E for escaped photons
    def __init__(self, n_ang, n_E, E_max, tot_n_part):

        self.n_ang = n_ang
        self.n_E = n_E
        self.E_max = E_max
        self.tot_n_part = tot_n_part
        self.delta_ang = math.pi / n_ang
        self.delta_E = E_max / n_E
        self.E_ab_hist = hist(n_E, E_max)
        self.ang_out_hist = hist(n_ang, math.pi)
        self.E_out_hist = hist(n_E, E_max)
        self.ang_bin = np.arange(self.delta_ang/2., math.pi, self.delta_ang)
        self.E_bin = np.arange(self.delta_E/2., self.E_max, self.delta_E)

    def add_count(self, output):
        hist_esc, E_ab, E, theta = output
        self.E_ab_hist.add_count(E_ab)
        if hist_esc:
            self.ang_out_hist.add_count(theta)
            self.E_out_hist.add_count(E)
            
    def plot(self):
        # canvas for plots
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

        # angular distribution of outgoing photons
        ang_out_hist = self.ang_out_hist.hist
        ax[0].bar(self.ang_bin / math.pi, ang_out_hist, width = self.delta_ang / math.pi)
        ax[0].set_title('Angular spectrum of outgoing photons')
        ax[0].set_xlabel(r'Angle ('+'\u03C0'+'$\cdot$'+'rad)')
        #ax[0].set_xlim(0., 1.)
        ax[0].set_ylabel('Number of photons')

        # energy distribution of outgoing photons
        E_out_hist = self.E_out_hist.hist
        ax[1].bar(self.E_bin, E_out_hist, width = self.delta_E)
        ax[1].set_title('Energy spectrum of outgoing photons')
        ax[1].set_xlabel('Energy (MeV)')
        #ax[1].set_xlim(0., self.E_max)
        ax[1].set_ylabel('Number of photons')

        # absorbed energy distribution
        E_ab_hist = self.E_ab_hist.hist       
        ax[2].bar(self.E_bin, E_ab_hist, width = self.delta_E)
        ax[2].set_title('Spectrum of absorbed energy')
        ax[2].set_xlabel('Energy (MeV)')
        #ax[2].set_xlim(0., self.E_max)
        ax[2].set_ylabel('Number of photons')
        ax[2].set_yscale('log')

    def ang_out(self):
        # angular distribution of outgoing photons
        ang_out_hist = self.ang_out_hist.hist
        ang_out_df = np.column_stack((self.ang_bin, ang_out_hist / self.tot_n_part))
        ang_out_df = pd.DataFrame(ang_out_df, columns = ['Angle/rad', 'photons/incid. gamma'])
        return ang_out_df

    def E_out(self):
        # energy distribution of outgoing photons
        E_out_hist = self.E_out_hist.hist
        E_out_df = np.column_stack((self.E_bin, E_out_hist / self.tot_n_part))
        E_out_df = pd.DataFrame(E_out_df, columns = ['Energy/MeV', 'photons/incid. gamma'])
        return E_out_df

    def E_ab(self):
        # absorbed energy distribution
        E_ab_hist = self.E_ab_hist.hist
        E_ab_df = np.column_stack((self.E_bin, E_ab_hist / self.tot_n_part))  
        E_ab_df = pd.DataFrame(E_ab_df, columns = ['Energy/MeV', 'photons/incid. gamma'])
        return E_ab_df

    def to_excel(self, fname):
        # excel file
        fname = fname + '.xlsx'
        hist_writer = pd.ExcelWriter(fname, engine='xlsxwriter')
        ang_out_df = self.ang_out()
        E_out_df = self.E_out()
        E_ab_df = self.E_ab()
        ang_out_df.to_excel(hist_writer, sheet_name='Ang_Spect_out_Photons')
        E_out_df.to_excel(hist_writer, sheet_name='En_Spect_out_Photons')
        E_ab_df.to_excel(hist_writer, sheet_name='Spect_abs_Energy')
        # hist_writer.save() # obsolete
        hist_writer.close()
        print(fname + ' written onto disk')

class fluence:
    # Fluence curve along z axis
    # E hist of photons flowing through ds as a function of z
    def __init__(self, geometry, n_z, n_E, E_max, tot_n_part):
        self.geometry = geometry
        self.n_z = n_z
        self.n_E = n_E
        self.E_max = E_max
        self.delta_E = E_max / n_E
        self.delta_r2 = geometry.voxelization.delta_r2
        self.tot_n_part = tot_n_part

        self.z = np.linspace(geometry.z_bott, geometry.z_top, n_z+1) # n_z intervals, but n_z+1 values including z_top
        self.fluence = np.zeros_like(self.z)
        self.hist = np.array([hist(n_E, E_max) for z in self.z]) # array of histograms
        self.E_bin = np.arange(self.delta_E/2., self.E_max, self.delta_E)

    def add_count(self, p_back, p_forw, step_length, E):
        select, cos_theta = self.flow(p_back, p_forw, step_length)
        counts = 1./cos_theta
        self.fluence[select] += counts
        for hist in self.hist[select]:
            hist.add_count(E, counts)
            
    def flow(self, p1, p2, l):
        # Check if the track intersects the z plane at a radius r such that r^2<delta_r2
        # z is an array
        # Calculate cos(theta) too
        f = np.zeros_like(self.z)
        f = f==1.
        z1 = p1[2]
        z2 = p2[2]
        dz = z2 - z1
        if dz==0.:
            return f, 0.
        
        between = (self.z>=min(z1,z2))&(self.z<=max(z1,z2))
        t = (self.z[between]-z1)/dz
        x = p1[0] + (p2[0]-p1[0])*t
        y = p1[1] + (p2[1]-p1[1])*t
        r2 = x*x + y*y
        f[between] = r2<=self.delta_r2
        return f, abs(dz/l)

    def plot(self):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        # Total fluence
        y = self.fluence / (math.pi*self.delta_r2) * 10000. / self.tot_n_part
        ymax = y.max() * 1.1
        ax[0].plot(self.z, y)
        ax[0].set_title('Normalized fluence')
        ax[0].set_ylim(ymin=0., ymax=ymax)
        ax[0].set_xlabel('z (cm)')
        ax[0].set_ylabel('m$^{-2}$')
        
        # Spectral fluence at the entrance (minimun z)
        y = self.hist[0].hist / (self.delta_E*math.pi*self.delta_r2) * 10000. / self.tot_n_part
        ax[1].bar(self.E_bin, y, width = self.delta_E)
        ax[1].set_title('z = ' + str(self.z[0]) +' cm')
        ax[1].set_xlabel('E (MeV)')
        ax[1].set_ylabel('MeV$^{-1}$·m$^{-2}$')

        # Spectral fluence at the exit (maximum z)
        y = self.hist[-1].hist / (self.delta_E*math.pi*self.delta_r2) * 10000. / self.tot_n_part
        ax[2].bar(self.E_bin, y, width = self.delta_E)
        ax[2].set_title('z = ' + str(self.z[-1]) +' cm')
        ax[2].set_xlabel('E (MeV)')
        ax[2].set_ylabel('MeV$^{-1}$·m$^{-2}$')

    def to_df(self):
        # Dataframe with spectral fluence data
        fluence_df = np.array([h.hist for h in self.hist])
        fluence_df = fluence_df / (self.delta_E * math.pi*self.delta_r2) * 10000. / self.tot_n_part
        fluence_df = pd.DataFrame(fluence_df, columns = np.round(self.E_bin, 4), index = self.z)
        fluence_df.index.name = 'z(cm)' # rows are z bins
        fluence_df.columns.name = 'E(MeV)' # columns are E bins

        # Total fluence is in the last column
        y = self.fluence / (math.pi*self.delta_r2) * 10000. / self.tot_n_part
        fluence_df['total'] = y
        return fluence_df

    def to_excel(self, fname):
        # excel file
        fname = fname + '.xlsx'
        #hist_writer = pd.ExcelWriter(fname, engine='xlsxwriter')
        fluence_df = self.to_df()
        open(fname, "w") # to excel file
        fluence_df.to_excel(fname, sheet_name='fluence', header='z(cm)', float_format='%.3e') # includes bining data
        #hist_writer.save()
        print(fname + ' written onto disk')
        