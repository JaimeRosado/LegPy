import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
import random
from .analysis_tools import ext_range
from .geometry import Cylinder, Sphere 


class esc_gammas:
    # Plot theta vs energy of escaped photons
    def __init__(self, E_max):
        self.E_max = E_max
        self.points = np.array([[0., 0.]])
        
    def add_count(self, E, theta):
        #hist_esc, E_ab, E, theta = output
        #if hist_esc:
            #self.points = np.append(self.points, [[theta, E]], axis = 0)
        self.points = np.append(self.points, [[theta, E]], axis = 0)
        
    def plot(self):
        points = self.points[1:] # delete inital [0,0]
        plt.figure()
        plt.scatter(points[:,0] / math.pi, points[:,1], marker = '.')
        plt.xlabel(r'Angle (π$\cdot$rad)')
        #FA warning plt.xlabel(r'Angle ('+'\u03C0'+'$\cdot$'+'rad)')
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

    def normalize(self, tot_n_part):
        return self.hist / tot_n_part / self.delta


class e_hists:
    # Histograms of final z and maximum z of electrons
    # Histogram of theta angle for backscattered electrons (z<0)
    def __init__(self, n_z, n_ang, z_top, part_type):
        self.n_z = n_z
        self.n_ang = n_ang
        self.z_top = z_top
        self.part_type = part_type
        self.range_hist = hist(n_z, z_top) # final z
        self.trans_hist = hist(n_z, z_top) # maximum z
        self.back_hist = hist(n_ang, math.pi, math.pi/2.) # theta of backscattered electrons
        self.delta_z = self.range_hist.delta
        self.delta_ang = self.back_hist.delta
        self.max_depth = 0.
        self.z_bin = np.arange(self.delta_z/2., self.z_top, self.delta_z)
        self.ang_bin = np.arange(math.pi/2. + self.delta_ang/2., math.pi, self.delta_ang)
        
    def add_count(self, e_in, z_max, position, theta):
        #e_in, E, z_max, position, theta = output
        z = position[2]
        self.range_hist.add_count(z)
        self.trans_hist.add_count(z_max)
        if z_max>self.max_depth:
            self.max_depth = z_max # Shown at the end of the simulation
        if not e_in and z<0.:
            self.back_hist.add_count(theta)

    def normalize(self, tot_n_part):
        self.range_hist = self.range_hist.normalize(tot_n_part)
        self.trans_hist = self.trans_hist.normalize(tot_n_part)
        self.back_hist = self.back_hist.normalize(tot_n_part)
        
    def plot(self):
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

        # Histogram of final z 
        range_coef = 1. - self.range_hist.cumsum() / self.range_hist.sum() # backscattered electrons excluded
        ax[0].bar(self.z_bin, self.range_hist, width = self.delta_z)
        ax[0].set_xlabel('Depth (cm)')
        ax[0].set_ylabel('cm$^{-1}$')
        ax[0].set_title(f'Range of {self.part_type}s')
        
        # Histogram of max z
        trans_coef = 1. - self.trans_hist.cumsum() / self.trans_hist.sum()  # backscattered electrons excluded
        print('Maximum depth (cm): ', round(self.max_depth, 3))
        ax[1].scatter(self.z_bin, trans_coef, s = 25)
        ax[1].set_xlabel('Depth (cm)')
        ax[1].set_ylabel(f'Fraction of {self.part_type}s')
        ax[1].set_title('Transmission coefficient')
        ax[1].set_xlim(xmin = 0.)
        ax[1].set_ylim(ymin = 0.)

        # Histogram of theta for backscattered electrons
        #back_hist_solid = back_hist / (2. * math.pi * np.sin(self.ang_bin))
        tot_back = self.back_hist.sum() * self.delta_ang
        print('Fraction of backscattered electrons: ', round(tot_back, 3))
        ax[2].bar(self.ang_bin, self.back_hist, width = self.delta_ang)
        ax[2].set_title(f'Angular spectrum of backscatered {self.part_type}s')
        ax[2].set_xlabel('Angle (rad)')
        #ax[2].set_xlim(0., 1.)
        ax[2].set_ylabel('rad$^{-1}$')

    def final_z(self):
        # Histogram of final z
        range_coef = 1. - self.range_hist.cumsum() / self.range_hist.sum() # backscattered electrons excluded
        range_df = np.column_stack((self.z_bin, self.range_hist, range_coef))
        range_df = pd.DataFrame(range_df, columns = ['z (cm)', 'cm$^{-1}$', 'fraction'])
        return range_df

    def max_z(self):
        # Histogram of max z
        trans_coef = 1. - self.trans_hist.cumsum() / self.trans_hist.sum()  # backscattered electrons excluded
        trans_df = np.column_stack((self.z_bin, self.trans_hist, trans_coef))
        trans_df = pd.DataFrame(trans_df, columns = ['z (cm)', 'cm$^{-1}$', 'fraction'])
        return trans_df

    def ext_range(self, definition="final"):
        if definition=="max":
            df = self.max_z()
        else:
            df = self.final_z()
        return ext_range(df)     

    def backscattering(self):
        # Histogram of theta for backscattered electrons
        back_hist_solid = self.back_hist / (2. * math.pi * np.sin(self.ang_bin))
        tot_back = self.back_hist.sum()
        back_df = np.column_stack((self.ang_bin, self.back_hist, back_hist_solid))
        back_df = pd.DataFrame(back_df, columns = ['angle (rad)', 'rad$^{-1}$', 'sr$^{-1}$'])
        return back_df


class Eab_hist:  #####JR revisar
    # Histogram of absorbed energy
    # Histograms of theta and E for escaped photons
    def __init__(self, n_E, E_max):
        self.n_E = n_E
        self.E_max = E_max
        self.E_ab_hist = hist(n_E, E_max)
        self.delta_E = self.E_ab_hist.delta
        self.E_bin = np.arange(self.delta_E/2., self.E_max, self.delta_E)

    def add_count(self, E_ab):
        if E_ab>0.:
            self.E_ab_hist.add_count(E_ab)

    def normalize(self, tot_n_part):
        self.E_ab_hist = self.E_ab_hist.normalize(tot_n_part)

    def plot(self):
        # absorbed energy distribution
        plt.figure()
        plt.bar(self.E_bin, self.E_ab_hist, width = self.delta_E)
        plt.xlabel('Energy (MeV)')
        plt.ylabel('MeV$^{-1}$')
        plt.title('Spectrum of absorbed energy')
        plt.grid(True, which = 'both')
        plt.yscale('log')

    def E_ab(self):
        # absorbed energy distribution
        E_ab_df = np.column_stack((self.E_bin, self.E_ab_hist))  
        E_ab_df = pd.DataFrame(E_ab_df, columns = ['Energy (MeV)', 'MeV$^{-1}$'])
        return E_ab_df

    def to_excel(self, fname):
        # excel file
        fname = fname + '.xlsx'
        hist_writer = pd.ExcelWriter(fname, engine='xlsxwriter')
        E_ab_df = self.E_ab()
        E_ab_df.columns = ['Energy (MeV)', 'MeV-1']
        E_ab_df.to_excel(hist_writer, sheet_name='Spect_abs_Energy')
        # hist_writer.save() # obsolete
        hist_writer.close()
        print(fname + ' written onto disk')
        
class gamma_hists:
    # Histograms of theta and E for escaped photons
    def __init__(self, n_ang, n_E, E_max):
        self.n_ang = n_ang
        self.n_ang_max = n_ang - 1
        self.n_E = n_E
        self.n_E_max = n_E - 1
        self.E_max = E_max
        self.delta_ang = math.pi / n_ang
        self.delta_E = E_max / n_E
        self.ang_hist_noint = np.zeros(n_ang)
        self.ang_hist_int = np.zeros(n_ang)
        self.E_hist_noint = np.zeros(n_E)
        self.E_hist_int = np.zeros(n_E)
        self.E_ang_hist_noint = np.zeros((n_E, n_ang))
        self.E_ang_hist_int = np.zeros((n_E, n_ang))
        self.ang_bin = np.arange(self.delta_ang/2., math.pi, self.delta_ang)
        self.E_bin = np.arange(self.delta_E/2., self.E_max, self.delta_E)

    def add_count(self, E, theta, interaction):
        ang_i = min(self.n_ang_max, int(theta / self.delta_ang))
        E_j = min(self.n_E_max, int(E / self.delta_E))
        if interaction:
            self.E_ang_hist_int[E_j, ang_i] += 1
        else:
            self.E_ang_hist_noint[E_j, ang_i] += 1

    def normalize(self, tot_n_part):
        self.E_ang_hist_noint /= (tot_n_part * self.delta_ang * self.delta_E)
        self.E_ang_hist_int /= (tot_n_part * self.delta_ang * self.delta_E)
        self.ang_hist_noint = self.E_ang_hist_noint.sum(axis=0) * self.delta_E
        self.ang_hist_int = self.E_ang_hist_int.sum(axis=0) * self.delta_E
        self.E_hist_noint = self.E_ang_hist_noint.sum(axis=1) * self.delta_ang
        self.E_hist_int = self.E_ang_hist_int.sum(axis=1) * self.delta_ang
            
    def plot(self, interaction):
        if interaction=='both':
            E_ang_hist = self.E_ang_hist_noint + self.E_ang_hist_int
            ang_hist = self.ang_hist_noint + self.ang_hist_int
            E_hist = self.E_hist_noint + self.E_hist_int
        elif not interaction or interaction=='False':
            E_ang_hist = self.E_ang_hist_noint
            ang_hist = self.ang_hist_noint
            E_hist = self.E_hist_noint
        else:
            E_ang_hist = self.E_ang_hist_int
            ang_hist = self.ang_hist_int
            E_hist = self.E_hist_int

        # canvas for plots
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

        # angular distribution of outgoing photons
        ax[0].bar(self.ang_bin, ang_hist, width = self.delta_ang)
        ax[0].set_title('Angular spectrum of outgoing photons')
        ax[0].set_xlabel('Angle (rad)')
        #ax[0].set_xlim(0., 1.)
        ax[0].set_ylabel('rad$^{-1}$')

        # energy distribution of outgoing photons
        ax[1].bar(self.E_bin, E_hist, width = self.delta_E)
        ax[1].set_title('Energy spectrum of outgoing photons')
        ax[1].set_xlabel('Energy (MeV)')
        #ax[1].set_xlim(0., self.E_max)
        ax[1].set_ylabel('MeV$^{-1}$')

        # Energy vs angle distribution
        extent = (0., math.pi, 0., self.E_max) # for imshow
        E_ang_log_hist = np.log10(E_ang_hist[::-1])
        vmax = E_ang_log_hist.max()
        psm = ax[2].imshow(E_ang_log_hist, vmax = vmax, extent = extent, aspect = 'auto')
        ax[2].set_title('Energy vs angle spectrum of outgoing photons')
        ax[2].set_xlabel('Angle (rad)')
        ax[2].set_ylabel('Energy (MeV)')
        # Color bar attached to last plot
        cbar = fig.colorbar(psm, aspect = 50., shrink = 0.65)
        cbar.ax.set_ylabel('log(rad$^{-1}$ MeV$^{-1}$)')

    def ang_out(self, interaction):
        if interaction=='both':
            ang_hist = self.ang_hist_noint + self.ang_hist_int
        elif not interaction or interaction=='False':
            ang_hist = self.ang_hist_noint
        else:
            ang_hist = self.ang_hist_int

        # angular distribution of outgoing photons
        ang_df = np.column_stack((self.ang_bin, ang_hist))
        ang_df = pd.DataFrame(ang_df, columns = ['Angle (rad)', 'rad$^{-1}$'])
        return ang_df

    def E_out(self, interaction):
        if interaction=='both':
            E_hist = self.E_hist_noint + self.E_hist_int
        elif not interaction or interaction=='False':
            E_hist = self.E_hist_noint
        else:
            E_hist = self.E_hist_int

        # energy distribution of outgoing photons
        E_df = np.column_stack((self.E_bin, E_hist))
        E_df = pd.DataFrame(E_df, columns = ['Energy (MeV)', 'MeV$^{-1}$'])
        return E_df

    def E_ang_out(self, interaction):
        if interaction=='both':
            E_ang_hist = self.E_ang_hist_noint + self.E_ang_hist_int
        elif not interaction or interaction=='False':
            E_ang_hist = self.E_ang_hist_noint
        else:
            E_ang_hist = self.E_ang_hist_int

        E_ang_df = pd.DataFrame(E_ang_hist, columns = self.ang_bin, index = self.E_bin)
        E_ang_df.index.name = 'Energy (MeV)'
        E_ang_df.columns.name = 'Angle (rad)'
        return E_ang_df

    def ang_out_to_excel(self, fname, interaction):
        # excel file
        fname = fname + '.xlsx'
        hist_writer = pd.ExcelWriter(fname, engine='xlsxwriter')
        ang_df = self.ang_out(interaction)
        ang_df.columns = ['Angle (rad)', 'rad-1']
        ang_df.to_excel(hist_writer, sheet_name='Ang_Spect_out_Photons')
        # hist_writer.save() # obsolete
        hist_writer.close()
        print(fname + ' written onto disk')

    def E_out_to_excel(self, fname, interaction):
        # excel file
        fname = fname + '.xlsx'
        hist_writer = pd.ExcelWriter(fname, engine='xlsxwriter')
        E_df = self.E_out(interaction)
        E_df.columns = ['Energy (MeV)', 'MeV-1']
        E_df.to_excel(hist_writer, sheet_name='E_Spect_out_Photons')
        # hist_writer.save() # obsolete
        hist_writer.close()
        print(fname + ' written onto disk')

    def to_excel(self, fname, interaction):
        # excel file
        fname = fname + '.xlsx'
        hist_writer = pd.ExcelWriter(fname, engine='xlsxwriter')
        E_ang_df = self.E_ang_out(interaction)
        E_ang_df.index.name = r'MeV \ rad'
        #FA warning E_ang_df.index.name = 'MeV \ rad'
        E_ang_df.to_excel(hist_writer, sheet_name='E_vs_Ang_Spect_out_Photons')
        # hist_writer.save() # obsolete
        hist_writer.close()
        print(fname + ' written onto disk')


class fluence_z: # fluence = 'z'
    # Fluence curve along z axis
    # E hist of photons flowing through ds as a function of z
    def __init__(self, geometry, n_z, n_E, E_max): 
        self.geometry = geometry
        self.n_E = n_E
        self.E_max = E_max
        self.delta_E = E_max / n_E
        self.delta_r2 = geometry.voxelization.delta_r2 
        
        self.z = np.linspace(geometry.z_bott, geometry.z_top, n_z+1) # n_z intervals, but n_z+1 values including z_top
        self.fluence = np.zeros_like(self.z)
        self.hist = np.array([hist(n_E, E_max) for z in self.z])
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
        f = np.zeros_like(self.fluence)
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

    def normalize(self, tot_n_part):
        area = math.pi * self.geometry.voxelization.delta_r2 
        norm = area / 10000. * tot_n_part 
        self.fluence /= norm
        norm *= self.delta_E
        self.hist = np.array([h.hist/norm for h in self.hist])
    
    def plot(self, **arg):
        # 'ri', 'xi' and 'ri' indexes only usable for fluence = True
        if arg.get('xi') is not None:
            print(f"Warning: unused index xi = {arg['xi']}")
        if arg.get('yi') is not None:
            print(f"Warning: unused index yi = {arg['yi']}")
        if arg.get('ri') is not None:
            print(f"Warning: unused index ri = {arg['ri']}")
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True) 
        # Total fluence.
        y = self.fluence
        ymax = y.max() * 1.1
        ax[0].plot(self.z, y)
        ax[0].set_title('Normalized fluence')
        ax[0].set_ylim(ymin=0., ymax=ymax)
        ax[0].set_xlabel('z (cm)')
        ax[0].set_ylabel('m$^{-2}$')
            
        # Spectral fluence at the entrance (minimun z). 
        y = self.hist[0,:]
        ax[1].bar(self.E_bin, y, width = self.delta_E)
        ax[1].set_title('z = ' + str(self.z[0]) +' cm')
        ax[1].set_xlabel('E (MeV)')
        ax[1].set_ylabel('MeV$^{-1}$·m$^{-2}$')
            
        # Spectral fluence at the exit (maximum z).
        y = self.hist[-1,:] 
        ax[2].bar(self.E_bin, y, width = self.delta_E)
        ax[2].set_title('z = ' + str(self.z[-1]) +' cm')
        ax[2].set_xlabel('E (MeV)')
        ax[2].set_ylabel('MeV$^{-1}$·m$^{-2}$')

    def to_df(self, **arg):
        # 'ri', 'xi' and 'ri' indexes only usable for fluence = True
        if arg.get('xi') is not None:
            print(f"Warning: unused index xi = {arg['xi']}")
        if arg.get('yi') is not None:
            print(f"Warning: unused index yi = {arg['yi']}")
        if arg.get('ri') is not None:
            print(f"Warning: unused index ri = {arg['ri']}")
        
        # Dataframe with spectral fluence data
        fluence_df = self.hist
        fluence_df = pd.DataFrame(fluence_df, columns = np.round(self.E_bin, 4), index = self.z)
        fluence_df.index.name = 'z(cm)' # rows are z bins
        fluence_df.columns.name = 'E(MeV)' # columns are E bins

        # Total fluence is in the last column
        y = self.fluence
        fluence_df['total'] = y
        return fluence_df

    def to_excel(self, fname, **arg):
        # 'ri', 'xi' and 'ri' indexes only usable for fluence = True
        if arg.get('xi') is not None:
            print(f"Warning: unused index xi = {arg['xi']}")
        if arg.get('yi') is not None:
            print(f"Warning: unused index yi = {arg['yi']}")
        if arg.get('ri') is not None:
            print(f"Warning: unused index ri = {arg['ri']}")
        
        # excel file
        fname = fname + '.xlsx'
        #hist_writer = pd.ExcelWriter(fname, engine='xlsxwriter')
        fluence_df = self.to_df()
        fluence_df.index.name = r'z(cm)\E(MeV)'
        #FA warning fluence_df.index.name = 'z(cm)\E(MeV)'
        open(fname, "w") # to excel file
        fluence_df.to_excel(fname, sheet_name='fluence', header='z(cm)', float_format='%.3e') # includes bining data.
        #hist_writer.save()
        print(fname + ' written onto disk')

class fluence_cyl: # fluence = True for cylindrical voxelization
    # Fluence curve along z and r axes
    # E hist of photons flowing through ds as a function of z
    def __init__(self, geometry, n_E, E_max):
        self.geometry = geometry
        self.n_E = n_E
        self.E_max = E_max
        self.i_max = n_E - 1
        self.delta_E = E_max / n_E

        n_r = geometry.voxelization.n_r
        self.r = np.linspace(0., geometry.r, n_r+1) # n_r intervals, but n_r+1 values including geometry.r
        self.rmin2 = self.r[:-1]**2
        self.rmax2 = self.r[1:]**2
        self.area = math.pi*(self.rmax2-self.rmin2) 
        self.rmin2[0] = -1.
        
        n_z = geometry.voxelization.n_z
        self.z = np.linspace(geometry.z_bott, geometry.z_top, n_z+1) # n_z intervals, but n_z+1 values including z_top
        self.fluence = np.zeros((n_z+1, n_r))
        self.hist = np.array([[hist(n_E, E_max) for r2 in self.rmax2] for z in self.z])
        #self.sp_fluence = np.zeros((n_z+1, n_r, n_E))
        self.E_bin = np.arange(self.delta_E/2., self.E_max, self.delta_E)

    def add_count(self, p_back, p_forw, step_length, E):
        select, cos_theta = self.flow(p_back, p_forw, step_length)
        counts = 1./cos_theta
        self.fluence[select] += counts 
        for hist in self.hist[select]:
            hist.add_count(E, counts)
        #i = min(self.i_max, int(E / self.delta_E))
        #self.sp_fluence[select, i] += counts
            
    def flow(self, p1, p2, l):
        # Check if the track intersects the z plane at a radius r such that r^2<delta_r2
        # z is an array
        # Calculate cos(theta) too
        f = np.zeros_like(self.fluence) 
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
        r2 = r2[:, np.newaxis]
        
        f[between, :] = (r2>self.rmin2)&(r2<=self.rmax2)
        
        return f, abs(dz/l) 

    def normalize(self, tot_n_part):
        norm = self.area / 10000. * tot_n_part
        self.fluence /= norm
        norm *= self.delta_E
        self.hist = np.array([[h.hist/n for h,n in zip(hist_z, norm)] for hist_z in self.hist])
        #self.sp_fluence /= norm
    
    def plot(self, ri, **arg):
        # 'xi' and 'yi' indexes only usable for cartesian voxelization
        if arg.get('xi') is not None:
            print(f"Warning: unused index xi = {arg['xi']}")
        if arg.get('yi') is not None:
            print(f"Warning: unused index yi = {arg['yi']}")
        
        # The r-axis coordinate of the plot is given by ri
        if ri is None:
            ri = 0
            print('Warning: no value provided for ri. Using default ri=0.')
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True) 
        # Total fluence.
        y = self.fluence[:,ri]
        ymax = y.max() * 1.1
        ax[0].plot(self.z, y)
        ax[0].set_title('Normalized fluence') 
        ax[0].set_ylim(ymin=0., ymax=ymax)
        ax[0].set_xlabel('z (cm)')
        ax[0].set_ylabel('m$^{-2}$')
            
        # Spectral fluence at the entrance (minimun z). 
        y = self.hist[0,ri,:]
        ax[1].bar(self.E_bin, y, width = self.delta_E)
        ax[1].set_title('z = ' + str(self.z[0]) +' cm')
        ax[1].set_xlabel('E (MeV)')
        ax[1].set_ylabel('MeV$^{-1}$·m$^{-2}$')
            
        # Spectral fluence at the exit (maximum z).
        y = self.hist[-1,ri,:] 
        ax[2].bar(self.E_bin, y, width = self.delta_E)
        ax[2].set_title('z = ' + str(self.z[-1]) +' cm')
        ax[2].set_xlabel('E (MeV)')
        ax[2].set_ylabel('MeV$^{-1}$·m$^{-2}$')

    def to_df(self, ri, **arg):
        # 'xi' and 'yi' indexes only usable for cartesian voxelization
        if arg.get('xi') is not None:
            print(f"Warning: unused index xi = {arg['xi']}")
        if arg.get('yi') is not None:
            print(f"Warning: unused index yi = {arg['yi']}")
        
        # The r-axis coordinate of the dataframe is given by ri
        if ri is None:
            ri = 0
            print('Warning: no value provided for ri. Using default ri=0.')

        fluence_df = self.hist[:,ri,:] 
        fluence_df = pd.DataFrame(fluence_df, columns = np.round(self.E_bin, 4), index = self.z)
        fluence_df.index.name = 'z(cm)' # rows are z bins
        fluence_df.columns.name = 'E(MeV)' # columns are E bins

        # Total fluence is in the last column
        y = self.fluence[:,ri]
        fluence_df['total'] = y
        return fluence_df

    def to_excel(self, fname, ri, **arg):
        # 'xi' and 'yi' indexes only usable for cartesian voxelization
        if arg.get('xi') is not None:
            print(f"Warning: unused index xi = {arg['xi']}")
        if arg.get('yi') is not None:
            print(f"Warning: unused index yi = {arg['yi']}")
        
        # The r-axis coordinate of the sheet is given by ri
        if ri is None:
            ri = 0
            print('Warning: no value provided for ri. Using default ri=0.')

        # excel file
        fname = fname + '.xlsx'
        #hist_writer = pd.ExcelWriter(fname, engine='xlsxwriter')
        fluence_df = self.to_df(ri)
        fluence_df.index.name = r'z(cm)\E(MeV)'
        #FA warning fluence_df.index.name = 'z(cm)\E(MeV)'
        open(fname, "w") # to excel file
        fluence_df.to_excel(fname, sheet_name='fluence %d' %ri, header='z(cm)', float_format='%.3e') # includes bining data.
        #hist_writer.save()
        print(fname + ' written onto disk')


class fluence_cart: # fluence = True for cartesian voxelization
    # Fluence curve throughout the space
    # E hist of photons flowing through ds as a function of z
    def __init__(self, geometry, n_E, E_max): 
        self.geometry = geometry
        self.n_E = n_E
        self.E_max = E_max
        self.delta_E = E_max / n_E
            
        self.n_x = geometry.voxelization.n_x 
        self.n_y = geometry.voxelization.n_y
        self.n_z = geometry.voxelization.n_z
        self.x = np.linspace(geometry.x_left, geometry.x_right, self.n_x+1) # n_x intervals, but n_x+1 values including x_right
        self.y = np.linspace(geometry.y_left, geometry.y_right, self.n_y+1) # n_y intervals, but n_y+1 values including y_right
        self.z = np.linspace(geometry.z_bott, geometry.z_top, self.n_z+1) # n_z intervals, but n_z+1 values including z_top
        self.area = geometry.voxelization.delta_x * geometry.voxelization.delta_y 

        # Define function to apply cut to photon track depending on geometry
        if isinstance(geometry, Cylinder): # Cylinder or sphere
            if isinstance(geometry, Sphere): # Sphere
                self.cut = self.cut_sph
            else:
                self.cut = self.cut_cyl
        else: # Orthoedron
            self.cut = self.cut_ortho
        
        self.xmin, self.ymin, self.xmax, self.ymax= self.x[:-1], self.y[:-1], self.x[1:], self.y[1:]
        self.fluence = np.zeros((self.n_z+1, self.n_x, self.n_y)) # Fluence has dimensions n_z+1 * n_x * n_y
        self.hist = np.array([[[hist(n_E, E_max) for y in self.ymin] for x in self.xmin] for z in self.z]) 
        self.E_bin = np.arange(self.delta_E/2., self.E_max, self.delta_E)
        
        self.xmin, self.xmax = self.xmin[np.newaxis,:,np.newaxis], self.xmax[np.newaxis,:,np.newaxis]
        self.ymin, self.ymax = self.ymin[np.newaxis,np.newaxis,:], self.ymax[np.newaxis,np.newaxis,:]

    def add_count(self, p_back, p_forw, step_length, E):
        select, cos_theta = self.flow(p_back, p_forw, step_length)
        counts = 1./cos_theta
        self.fluence[select] += counts 
        for hist in self.hist[select]:
            hist.add_count(E, counts)
            
    def flow(self, p1, p2, l):
        # Check if the track intersects the z plane in a coordinate (x,y) such that xmin<x<=xmax and ymin<y<=ymax
        # z is an array
        # Calculate cos(theta) too
        f = np.zeros_like(self.fluence) 
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
        x, y = self.cut(between, x, y)

        x = x[:, np.newaxis, np.newaxis]
        y = y[:, np.newaxis, np.newaxis]
        
        f[between,:,:] = (((x>self.xmin)&(x<=self.xmax))&((y>self.ymin)&(y<=self.ymax)))
 
        return f, abs(dz/l)

    def normalize(self,tot_n_part):
        norm = self.area / 10000. * tot_n_part
        self.fluence /= norm
        norm *= self.delta_E
        self.hist = np.array([[[self.hist[z,x,y].hist / norm for y in range(self.n_y)] for x in range(self.n_x)] for z in range(self.n_z+1)])
        
    def plot(self, xi, yi, **arg):
        # 'ri' index only usable for cylindrical voxelization
        if arg.get('ri') is not None:
            print(f"Warning: unused index ri = {arg['ri']}")
        
        # The XY plane coordinates of the plot are given by xi and yi
        if xi is None:
            xi = self.n_x//2
            print(f'Warning: no value provided for xi. Using default xi={xi}.')
            if self.n_x%2==0: 
                print('Warning: default x value is not at the center (even n_x).') 
        if yi is None:
            yi = self.n_y//2
            print(f'Warning: no value provided for yi. Using default yi={yi}.') 
            if self.n_y%2==0:
                print('Warning: default y value is not at the center (even n_y).')
        
        if isinstance(self.geometry, Cylinder) or isinstance(self.geometry, Sphere): # For non-Cartesian symmetries
            if self.x[xi]**2 + self.y[yi]**2 > self.geometry.r2:
                print('Warning: indexes out of medium.')
        
        fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True) 
        # Total fluence.
        y = self.fluence[:,xi,yi] 
        ymax = y.max() * 1.1
        ax[0].plot(self.z, y)
        ax[0].set_title('Normalized fluence')
        ax[0].set_ylim(ymin=0., ymax=ymax)
        ax[0].set_xlabel('z (cm)')
        ax[0].set_ylabel('m$^{-2}$')
            
        # Spectral fluence at the entrance (minimun z). 
        y = self.hist[0,xi,yi,:]
        ax[1].bar(self.E_bin, y, width = self.delta_E)
        ax[1].set_title('z = ' + str(self.z[0]) +' cm')
        ax[1].set_xlabel('E (MeV)')
        ax[1].set_ylabel('MeV$^{-1}$·m$^{-2}$')
            
        # Spectral fluence at the exit (maximum z).
        y = self.hist[-1,xi,yi,:]
        ax[2].bar(self.E_bin, y, width = self.delta_E)
        ax[2].set_title('z = ' + str(self.z[-1]) +' cm')
        ax[2].set_xlabel('E (MeV)')
        ax[2].set_ylabel('MeV$^{-1}$·m$^{-2}$')

    def to_df(self, xi, yi, **arg):
        # 'ri' index only usable for cylindrical voxelization
        if arg.get('ri') is not None:
            print(f"Warning: unused index ri = {arg['ri']}")
        
        # The XY plane coordinates of the dataframe are given by xi and yi
        if xi is None:
            xi = self.n_x//2
            print(f'Warning: no value provided for xi. Using default xi={xi}.')
            if self.n_x%2==0: 
                print('Warning: default x value is not at the center (even n_x).') 
        if yi is None:
            yi = self.n_y//2
            print(f'Warning: no value provided for yi. Using default yi={yi}.') 
            if self.n_y%2==0:
                print('Warning: default y value is not at the center (even n_y).')

        # Dataframe with spectral fluence data
        fluence_df = self.hist[:,xi,yi,:] 
        fluence_df = pd.DataFrame(fluence_df, columns = np.round(self.E_bin, 4), index = self.z)
        fluence_df.index.name = 'z(cm)' # rows are z bins
        fluence_df.columns.name = 'E(MeV)' # columns are E bins

        # Total fluence is in the last column
        y = self.fluence[:,xi,yi]
        fluence_df['total'] = y
        return fluence_df

    def to_excel(self, fname, xi, yi, **arg):
        # 'ri' index only usable for cylindrical voxelization
        if arg.get('ri') is not None:
            print(f"Warning: unused index ri = {arg['ri']}")
        
        # The XY plane coordinates of the sheet are given by xi and yi
        if xi is None:
            xi = self.n_x//2
            print(f'Warning: no value provided for xi. Using default xi={xi}.')
            if self.n_x%2==0: 
                print('Warning: default x value is not at the center (even n_x).') 
        if yi is None:
            yi = self.n_y//2
            print(f'Warning: no value provided for yi. Using default yi={yi}.') 
            if self.n_y%2==0:
                print('Warning: default y value is not at the center (even n_y).')
                
        # excel file
        fname = fname + '.xlsx'
        #hist_writer = pd.ExcelWriter(fname, engine='xlsxwriter')
        fluence_df = self.to_df(xi,yi)
        fluence_df.index.name = r'z(cm)\E(MeV)'
        #FA warning fluence_df.index.name = 'z(cm)\E(MeV)'
        open(fname, "w") # to excel file
        fluence_df.to_excel(fname, sheet_name='fluence'+str((xi,yi)), header='z(cm)', float_format='%.3e') # includes bining data.
        #hist_writer.save()
        print(fname + ' written onto disk')

    def cut_ortho(self, between, x, y):
        # Do nothing
        return x, y

    def cut_cyl(self, between, x, y): 
        # Apply cut to the photon track for cylindrical geometry
        r2 = x**2 + y**2 
        cut = r2>self.geometry.r2
        x[cut] = np.nan 
        y[cut] = np.nan
        return x, y

    def cut_sph(self, between, x, y):
        # Apply cut to the photon track for spherical geometry
        r2 = x**2 + y**2 + self.z[between]**2
        cut = r2>self.geometry.r2
        x[cut] = np.nan
        y[cut] = np.nan
        return x, y
