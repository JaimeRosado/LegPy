# Author: Fernando Arqueros

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math 

## Cross sections

def txsect_KN(gamma_energy):
    # Total KN cross section. For normalization of angular distribution AD_KN
    gamma = gamma_energy / 0.511 #electron mass
    r0 = 2.817e-13
    cs_kn = ((1.+gamma)/gamma/gamma*(2.*(1.+gamma)/(1.+2.*gamma)-(math.log(1.+2.*gamma)/gamma)
            )+math.log(1.+2.*gamma)/(2.*gamma)-(1.+3.*gamma)/(1.+2.*gamma)**2
            )*2.*math.pi*r0*r0
    return cs_kn

def axsect_KN(gamma_energy, theta):
    # KN dif xsection (angular). sigma(theta) 
    r0 = 2.817e-13
    g = gamma_energy / 0.511
    # s_kn = txsect_KN(Energy) 
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    dsda_KN = r0**2 * math.pi * sin_t / (1. + g * (1. - cos_t))**2 * (1. + cos_t**2 + (g**2 * (1.-cos_t)**2 / 
                (1. + g * (1. - cos_t))))
    return dsda_KN

def axsect_Ry(theta): 
    # dif xsection (angular)  of Coherent scattering P(theta); aproximation by scatt. Thomson law
    ad_Ry = 3./8. * np.sin(theta)*(1.+np.cos(theta)**2)
    return ad_Ry

def exsect_KN(gamma_energy, electron_energy):
    # KN dif xsection (electron energy). sigma(Te) 
    r0 = 2.817e-13
    mcc = 0.511
    g = gamma_energy / mcc
    s = electron_energy / mcc
    S = s / g
    dsdT_KN = math.pi*r0*r0/mcc/g/g*(2.+S*S/g/g/(1.-S)**2+S/(1.-S)*(S-2./g)) # [cm^2 / MeV]
    return dsdT_KN

def gaussian(x, mu, sig): # normalized
    return 1. / np.power(2. * np.pi, 0.5) / sig * np.exp(-np.power(x - mu, 2.) / (2. * np.power(sig, 2.)))

## Convolution product
#def convolution(in_spect_df, sigma):
def convolution(in_spect, sigma):

    #in_spect = in_spect_df.to_numpy()
    xx = in_spect[:,0] # array of energies (x) in in_spect
    yy = in_spect[:,1] # array of intensities (y) in_spect
 
    Delta_x = xx[-1] - xx[0] # size of the x-range of input
    delta_x = (Delta_x/len(xx)) # bin size of the x input
    delta_x_2 = xx[1] - xx[0]
    n = int(5*sigma/delta_x) # extra bins needed at the end to extend the profile (delta -> gaussian)
    xxc = xx # x bins for output
    yyc = yy # y bins for output
    ad = delta_x * np.ones(n)
    for add in ad: 
        xxc = np.append(xxc, add + xxc[-1]) # extended
        yyc = np.append(yyc, 0.) # y = 0 at the end (extra bins)
 
    y_out = np.zeros_like(yyc) # initilization of output array

    for i in range(len(xxc)): 
        for j in range(len(xxc)):
            #y_out[i] = y_out[i] + yyc[j] * gaussian(xxc[i], xxc[j], sigma/xx[-1] * xxc[j]) * delta_x # convolution product
            y_out[i] = y_out[i] + yyc[j] * gaussian(xxc[i], xxc[j], sigma) * delta_x_2 # convolution product

    out_spect = np.column_stack((xxc, y_out))
    out_spect_df = pd.DataFrame(out_spect, columns = ['Energy/MeV', 'photons/incid. gamma'])

    return out_spect_df


## Extrapolated range
def ext_range(hist_df):
    # T(z) plot
    plt.figure()
    z_bin = hist_df.iloc[:,0]
    delta_z = z_bin[1] - z_bin[0]
    hist = hist_df.iloc[:,1]
    n_part = hist.sum()
    T = hist_df.iloc[:,2]
    plt.scatter(z_bin, T, marker=".", c="k")
    i = hist.idxmax()
    z_tp, hist_tp, T_tp = hist_df.iloc[i]
    x = [z_tp+T_tp/hist_tp*n_part*delta_z ,  z_tp-(1-T_tp)/hist_tp*n_part*delta_z]
    R_ext = x[0]
    y = [0., 1.]
    plt.plot(x,y)
    plt.xlabel('z_depth (cm)')
    plt.ylabel('Transmission')
    plt.grid()
    plt.show()
    
    z_av = (hist*z_bin).sum() / n_part # Average depth
    print('Extrapolated range (cm): ', round(R_ext, 3))
    print('Distribution mode (cm): ', round(z_tp, 3))
    print('Distribution average (cm): ', round(z_av, 3))
    return R_ext, z_tp, z_av

def plot_edep_z(E_dep, x_size, y_size, z_size, n_z, n_x, n_y, z_ind, c_profiles = False, lev = None): # plot of E_dep in z (depth) layers
    # z_ind is an array of indexes of z layers ti bo plot; e.g.,  ([0, 6, 12, 18, 24])
    X = np.linspace(-x_size/2., x_size/2., n_x) # for contours
    Y = np.linspace(-y_size/2., y_size/2., n_y) # for contours
    E_dep_max = np.amax(E_dep) # for normalization
    # 
    fig, ax = plt.subplots(1, 5, figsize=(15, 4), constrained_layout=True)
    extent = (-x_size / 2., x_size / 2., -y_size / 2., y_size / 2.) # for imshow
    delta_z = z_size / n_z
    im = 0
    for z in z_ind:
        E_dep_log_z = np.log10(np.transpose(E_dep[:,:,z])[::-1]) # log(E_dep) of iz layer [:,:,iz], transposed and rotated along the x-axis [::-1] to match the image xy axes
        psm = ax[im].imshow(E_dep_log_z, vmax = np.log10(E_dep_max), extent=extent)
        z_plot = (z + 0.5) * delta_z # center of z bin
        ax[im].set_title('z = ' + str(round(z_plot,3)) +' cm')
        ax[im].set_xlabel('x (cm)')
        ax[im].set_ylabel('y (cm)')
        if c_profiles:
            CS = ax[im].contour(X,Y, E_dep_log_z, lev, colors='black', linestyles='solid')
            ax[im].clabel(CS, inline=True, fontsize=7, fmt = '%1.1f')
        im +=1

    # Color bar attached to last plot
    cbar = fig.colorbar(psm, aspect = 50., shrink = 0.65)
    cbar.ax.set_ylabel('log(E$_{dep}$ (keV cm$^{-3}$))')
    # 
    #plt.show()
    return    

def plot_edep_y(E_dep, x_size, y_size, z_size, n_z, n_x, n_y, y_ind, c_profiles = False, lev = None): # plot of E_dep in y layers
    # y_ind is an array of indexes of y layers ti bo plot; e.g.,  ([0, 6, 12, 18, 24])
    X = np.linspace(-x_size/2., x_size/2., n_x) # for contours
    Z = np.linspace(0., z_size, n_z) # for contours
    E_dep_max = np.amax(E_dep) # for normalization
    #
    fig, ax = plt.subplots(1, 5, figsize=(15, 4), constrained_layout=True)
    extent = (0., z_size, -x_size / 2., x_size / 2.)
    delta_y = y_size / n_y
    im = 0

    for y in y_ind:
        E_dep_log_z = np.log10((E_dep[:,y,:])) # log(E_dep) of iz layer [:,:,iz], transposed and rotated along the x-axis [::-1] to match the image xy axes
        psm = ax[im].imshow(E_dep_log_z, vmax = np.log10(E_dep_max), extent=extent)
        y_plot = (y + 0.5) * delta_y - y_size/2 # center of y bin
        ax[im].set_title('y = ' + str(round(y_plot,3)) +' cm')
        ax[im].set_xlabel('z (cm)')
        ax[im].set_ylabel('x (cm)')
        if c_profiles:
            CS = ax[im].contour(Z,X, E_dep_log_z, lev, colors='black', linestyles='solid')
            ax[im].clabel(CS, inline=True, fontsize=7, fmt = '%1.1f')
        im +=1
    
    # Color bar attached to last plot
    cbar = fig.colorbar(psm, aspect = 50., shrink = 0.65)
    cbar.ax.set_ylabel('log(E$_{dep}$ (keV cm$^{-3}$))')
    #plt.show()
    return    

def plot_edep_x(E_dep, x_size, y_size, z_size, n_z, n_x, n_y, x_ind, c_profiles = False, lev = None): # plot of E_dep in x layers
    # x_ind is an array of indexes of z layers ti bo plot; e.g.,  ([0, 6, 12, 18, 24])
    Y = np.linspace(-y_size/2., y_size/2., n_y) # for contours
    Z = np.linspace(0., z_size, n_z) # for contours
    E_dep_max = np.amax(E_dep) # for normalization
    #
    fig, ax = plt.subplots(1, 5, figsize=(15, 4), constrained_layout=True)
    extent = (0, z_size, -y_size / 2., y_size / 2.)
    delta_x = x_size / n_x

    im = 0
    for x in x_ind:
        E_dep_log_z = np.log10((E_dep[x,:,:])) # log(E_dep) of iz layer [:,:,iz], transposed and rotated along the x-axis [::-1] to match the image xy axes
        psm = ax[im].imshow(E_dep_log_z, vmax = np.log10(E_dep_max), extent=extent)
        x_plot = (x + 0.5) * delta_x - x_size/2 # center of y bin
        ax[im].set_title('x = ' + str(round(x_plot,3)) +' cm')
        ax[im].set_xlabel('z (cm)')
        ax[im].set_ylabel('y (cm)')
        if c_profiles:
            CS = ax[im].contour(Z,Y, E_dep_log_z, lev, colors='black', linestyles='solid')
            ax[im].clabel(CS, inline=True, fontsize=7, fmt = '%1.1f')
        im +=1
    
    # Color bar attached to last plot
    cbar = fig.colorbar(psm, aspect = 50., shrink = 0.65)
    cbar.ax.set_ylabel('log(E$_{dep}$ (keV cm$^{-3}$))')

    #plt.show()
    return

# Evaluates the integral of the product f1(x) * f2(x) within the limits xi, xf
def int_pro(f1, f2, xi, xf):
    if np.size(f1) >= np.size(f2):
        x1 = f1[:,0]
        y1 = f1[:,1]
        x2 = f2[:,0]
        y2 = f2[:,1]
        
    else:
        x1 = f2[:,0]
        y1 = f2[:,1]
        x2 = f1[:,0]
        y2 = f1[:,1]
    
    x1_low = f1[:,0].min()
    x2_low = f2[:,0].min()
    x1_up = f1[:,0].max()
    x2_up = f2[:,0].max()
    
    if (x1_low > xi) or (x2_low > xi) or (x1_up < xf) or (x2_up < xf):
        return print('check consistency between limits of integral and data')
    
    xlim = x1[(x1 >= xi)&(x1 <= xf)]
    xlim = np.insert(xlim, 0, xi)
    xlim = np.insert(xlim, xlim.size, xf)
    
    y1lim = np.interp(xlim, x1, y1)
    y2lim = np.interp(xlim, x2, y2)
    
    interv = (np.roll(xlim,-1) - xlim)[:-1] # x intervals
    mid_x = xlim[:-1] + interv/2 # mid x-point
    
    integr = np.interp(mid_x, xlim, y1lim * y2lim)

    I = np.sum(integr * interv)
    
    return I
    


