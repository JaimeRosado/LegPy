# Author: Fernando Arqueros

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
import random


class esc_gammas:
    # Plot theta vs energy of escaped photons
    def __init__(self, E_max):
        self.E_max = E_max
        self.plot = np.array([[0., 0.]])
        
    def add_count(self, output):
        hist_esc, E_ab, E, theta = output
        if hist_esc:
            self.plot = np.append(self.plot, [[theta, E]], axis = 0)
        
    def out(self):
        plot = self.plot[1:] # delete inital [0,0]
        plt.figure()
        plt.scatter(plot[:,0] / math.pi, plot[:,1], marker = '.')
        plt.xlabel('Angle ('+'\u03C0'+'$\cdot$'+'rad)')
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

    def add_count(self, val):
        if val<self.val_min:
            self.left += 1
        elif val>self.val_max:
            self.right +=1
        else:
            val = val - self.val_min
            i = min(self.i_max, int(val / self.delta))
            self.hist[i] += 1


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
        
    def add_count(self, output):
        e_in, E, z_max, position, theta = output
        z = position[2]
        self.range_hist.add_count(z)
        self.trans_hist.add_count(z_max)
        if z_max>self.max_depth:
            self.max_depth = z_max # Shown at the end of the simulation
        if not e_in and z<0.:
            self.back_hist.add_count(theta)
        
    def out(self, h_save, h_plot):
        if h_plot:
            fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

        # Histogram of final z
        z_bin = np.arange(self.delta_z/2., self.z_top, self.delta_z)
        range_hist = self.range_hist.hist
        range_coef = 1. - self.range_hist.hist.cumsum() / self.range_hist.hist.sum() # backscattered electrons excluded
        if h_plot:
            ax[0].bar(z_bin, range_hist, width = self.delta_z)
            ax[0].set_xlabel('Depth (cm)')
            ax[0].set_ylabel('Number of electrons')
            ax[0].set_title('Range of electrons')
        range_df = np.column_stack((z_bin, range_hist, range_coef))
        range_df = pd.DataFrame(range_df, columns = ['z/cm', 'electrons', 'fraction'])
        
        # Histogram of max z
        trans_hist = self.trans_hist.hist
        trans_coef = 1. - self.trans_hist.hist.cumsum() / self.trans_hist.hist.sum()  # backscattered electrons excluded
        print('Maximum depth (cm): ', round(self.max_depth, 3))
        if h_plot:
            ax[1].scatter(z_bin, trans_coef, s = 25)
            ax[1].set_xlabel('Depth (cm)')
            ax[1].set_ylabel('Fraction of electrons')
            ax[1].set_title('Transmission coefficient')
            ax[1].set_xlim(xmin = 0.)
            ax[1].set_ylim(ymin = 0.)
        trans_df = np.column_stack((z_bin, trans_hist, trans_coef))
        trans_df = pd.DataFrame(trans_df, columns = ['z/cm', 'electrons', 'fraction'])

        # Histogram of theta for backscattered electrons
        ang_bin = np.arange(math.pi/2. + self.delta_ang/2., math.pi, self.delta_ang)
        back_hist = self.back_hist.hist
        back_hist_solid = back_hist / self.delta_ang / self.tot_n_part / (2. * math.pi * np.sin(ang_bin))
        tot_back = back_hist.sum()
        print('Fraction of backscattered electrons: ', round(tot_back/self.tot_n_part, 3))
        if h_plot:
            ax[2].bar(ang_bin / math.pi, back_hist, width = self.delta_ang / math.pi)
            ax[2].set_title('Angular spectrum of backscatered electrons')
            ax[2].set_xlabel('Angle ('+'\u03C0'+'$\cdot$'+'rad)')
            #ax[2].set_xlim(0., 1.)
            ax[2].set_ylabel('Number of electrons')
        back_df = np.column_stack((ang_bin, back_hist, back_hist_solid))
        back_df = pd.DataFrame(back_df, columns = ['angle/rad', 'electrons', 'dn/dOmega'])

        return (range_df, trans_df, back_df)


class gamma_hists:
    # Histogram of absorbed energy
    # Histograms of theta and E for escaped photons
    def __init__(self, medium, n_ang, n_E, E_max, tot_n_part):

        self.medium = medium
        self.n_ang = n_ang
        self.n_E = n_E
        self.E_max = E_max
        self.tot_n_part = tot_n_part
        self.delta_ang = math.pi / n_ang
        self.delta_E = E_max / n_E
        self.E_ab_hist = hist(n_E, E_max)
        self.ang_out_hist = hist(n_ang, math.pi)
        self.E_out_hist = hist(n_E, E_max)

    def add_count(self, output):
        hist_esc, E_ab, E, theta = output
        self.E_ab_hist.add_count(E_ab)
        if hist_esc:
            self.ang_out_hist.add_count(theta)
            self.E_out_hist.add_count(E)
            
    def out(self, h_save, h_plot):
        # canvas for plots
        if h_plot:
            fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

        # angular distribution of outgoing photons
        ang_bin = np.arange(self.delta_ang/2., math.pi, self.delta_ang)
        ang_out_hist = self.ang_out_hist.hist

        # plot
        if h_plot:
            ax[0].bar(ang_bin / math.pi, ang_out_hist, width = self.delta_ang / math.pi)
            ax[0].set_title('Angular spectrum of outgoing photons')
            ax[0].set_xlabel('Angle ('+'\u03C0'+'$\cdot$'+'rad)')
            #ax[0].set_xlim(0., 1.)
            ax[0].set_ylabel('Number of photons')

        # output
        ang_out_df = np.column_stack((ang_bin, ang_out_hist / self.tot_n_part))
        ang_out_df = pd.DataFrame(ang_out_df, columns = ['Angle/rad', 'photons/incid. gamma'])

        # energy distribution of outgoing photons
        E_bin = np.arange(self.delta_E/2., self.E_max, self.delta_E)
        E_out_hist = self.E_out_hist.hist

        # plot
        if h_plot:
            ax[1].bar(E_bin, E_out_hist, width = self.delta_E)
            ax[1].set_title('Energy spectrum of outgoing photons')
            ax[1].set_xlabel('Energy (MeV)')
            #ax[1].set_xlim(0., self.E_max)
            ax[1].set_ylabel('Number of photons')

        # output
        E_out_df = np.column_stack((E_bin, E_out_hist / self.tot_n_part))
        E_out_df = pd.DataFrame(E_out_df, columns = ['Energy/MeV', 'photons/incid. gamma'])

        # absorbed energy distribution
        E_ab_hist = self.E_ab_hist.hist

        # plot
        if h_plot:        
            ax[2].bar(E_bin, E_ab_hist, width = self.delta_E)
            ax[2].set_title('Spectrum of absorbed energy')
            ax[2].set_xlabel('Energy (MeV)')
            #ax[2].set_xlim(0., self.E_max)
            ax[2].set_ylabel('Number of photons')
            ax[2].set_yscale('log')

        # output
        E_ab_df = np.column_stack((E_bin, E_ab_hist / self.tot_n_part))  
        E_ab_df = pd.DataFrame(E_ab_df, columns = ['Energy/MeV', 'photons/incid. gamma'])

        # # excel files
        if h_save:
            En = "{:.{}f}".format( self.E_max, 2 ) + 'MeV'
            m_type = self.medium.name
            excel_name = 'ang_ener_'+ m_type + '_' + En + '.xlsx'
            hist_writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')
            ang_out_df.to_excel(hist_writer, sheet_name = 'Ang_Spect_out_Photons')
            E_out_df.to_excel(hist_writer, sheet_name = 'En_Spect_out_Photons')
            E_ab_df.to_excel(hist_writer, sheet_name = 'Spect_abs_Energy')
            hist_writer.save()
            print(excel_name + ' written onto disk')

        return (ang_out_df, E_out_df, E_ab_df)
    
class fluence_data:
    # Fluence curve along z axis
    # E hist of photons flowing through ds as a function of z
    def __init__(self, geometry, medium, n_z, n_E, E_max):
        self.geometry = geometry
        self.n_z = n_z
        self.n_E = n_E
        self.E_max = E_max
        self.delta_E = E_max / n_E
        self.delta_r2 = geometry.voxelization.delta_r2

        self.z = np.linspace(geometry.z_bott, geometry.z_top, n_z+1) # n_z intervals, but n_z+1 values including z_top
        self.fluence = np.zeros_like(self.z)
        self.hist = np.array([hist(n_E, E_max) for z in self.z]) # array of histograms
        
        self.medium = medium
    def add_count(self, p_back, p_forw, step_length, E):
        select, cos_theta = self.flow(p_back, p_forw, step_length)
        self.fluence[select] += 1./cos_theta
        for hist in self.hist[select]:
            hist.add_count(E)
            
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

        
    def out(self, n_part, h_save, h_plot):
        y = self.fluence / (math.pi*self.delta_r2) * 10000. / n_part

        if h_plot:
            fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
            # fluence curve
            ymax = y.max() * 1.1
            ax[0].plot(self.z, y)
            ax[0].set_title('Normalized fluence')
            ax[0].set_ylim(ymin=0., ymax=ymax)
            ax[0].set_xlabel('z (cm)')
            ax[0].set_ylabel('m$^{-2}$')

        E_bin = np.arange(self.delta_E/2., self.E_max, self.delta_E)
        fluence_df = np.array([h.hist for h in self.hist])
        fluence_df = fluence_df / (self.delta_E * math.pi*self.delta_r2) * 10000. / n_part
        fluence_df = pd.DataFrame(fluence_df, columns = np.round(E_bin, 2), index = self.z)
        fluence_df.index.name = 'z(cm)'
        fluence_df.columns.name = 'E(MeV)'
        
        if h_plot:
            E_hist = fluence_df.iloc[0,:]
            ax[1].bar(E_bin, E_hist, width = self.delta_E)
            ax[1].set_title('z = ' + str(self.z[0]) +' cm')
            ax[1].set_xlabel('E (MeV)')
            ax[1].set_ylabel('MeV$^{-1}$·m$^{-2}$')

            E_hist = fluence_df.iloc[-1,:]
            ax[2].bar(E_bin, E_hist, width = self.delta_E)
            ax[2].set_title('z = ' + str(self.z[-1]) +' cm')
            ax[2].set_xlabel('E (MeV)')
            ax[2].set_ylabel('MeV$^{-1}$·m$^{-2}$')

        fluence_df['total'] = y
        
        if h_save:
            En = "{:.{}f}".format( self.E_max, 2 ) + 'MeV'
            m_type = self.medium.name
            excel_name = 'fluence_'+ m_type + '_' + En + '.xlsx'
            hist_writer = pd.ExcelWriter(excel_name, engine='xlsxwriter')
            open(excel_name, "w") # to excel file
            fluence_df.to_excel(excel_name, sheet_name = 'fluence', header = 'z(cm)', float_format = '%.3e') # includes bining data

        return fluence_df