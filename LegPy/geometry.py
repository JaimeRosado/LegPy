# Author: Fernando Arqueros

import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, PathPatch
import mpl_toolkits.mplot3d.art3d as art3d
import warnings
warnings.filterwarnings(
    'ignore',
    'divide by zero encountered in log10',
    RuntimeWarning)
warnings.filterwarnings(
    'ignore',
    'Warning: converting a masked element to nan.',
    UserWarning)

def Geometry(name='cylinder', x=None, y=None, z=None, r=None, diam=None,
             n_x=None, n_y=None, n_z=None, n_r=None,
             z_ch=None, r_ch=None):

    if name == 'orthohedron':
        if x is None or y is None or z is None:
            raise ValueError('Please, input x, y, z.')
        if n_x is None or n_y is None or n_z is None:
            raise ValueError('Please, input n_x, n_y and n_z.')
        if x<=0. or y<=0. or z<=0.:
            raise ValueError('x, y, z should be greater than 0.')
        if z_ch is not None:
            if z_ch<=0. or z_ch>=z:
                raise ValueError('z_ch should be between 0 and z.')
        # Cartesian voxelization
        geometry = Ortho(x, y, z, z_ch)
        voxelization = cart_vox(geometry, n_x, n_y, n_z)

    elif name == 'cylinder':
        if (r is None and diam is None) or z is None:
            raise ValueError('Please, input r or diam and z.')
        if diam is not None:
            r = diam / 2.
        if r<=0. or z<=0.:
            raise ValueError('r and z should be greater than 0.')
        if z_ch is not None:
            if z_ch<=0. or z_ch>=z:
                raise ValueError('z_ch should be between 0 and z.')
        elif r_ch is not None:
            if r_ch<=0. or r_ch>=r:
                raise ValueError('r_ch should be between 0 and r.')
        if n_x is not None and n_y is not None and n_z is not None: # Cartesian voxelization
            geometry = Cylinder(r, z, z_ch, r_ch)
            voxelization = cart_vox(geometry, n_x, n_y, n_z)
        elif n_z is not None and n_r is not None: # Cylindrical voxelization
            geometry = Cylinder(r, z, z_ch, r_ch)
            voxelization = cyl_vox(geometry, n_z, n_r)
        else:
            raise ValueError('Please, input either n_x, n_y and n_z or n_z and n_r.')

    elif name == 'sphere':
        if r is None and diam is None:
            raise ValueError('Please, input r or diam.')
        if diam is not None:
            r = diam / 2.
        if r<=0.:
            raise ValueError('r should be greater than 0.')
        if z_ch is not None:
            if z_ch<=-r or z_ch>=r:
                raise ValueError('z_ch should be between -r and r.')
        elif r_ch is not None:
            if r_ch<=0. or r_ch>=r:
                raise ValueError('r_ch should be between 0 and r.')
        if n_x is not None and n_y is not None and n_z is not None: # Cartesian voxelization
            geometry = Sphere(r, z_ch, r_ch)
            voxelization = cart_vox(geometry, n_x, n_y, n_z)
        elif n_r is not None: # Spherical voxelization
            geometry = Sphere(r, z_ch, r_ch)
            voxelization = sph_vox(geometry, n_r)
        else:
            raise ValueError('Please, input either n_x, n_y and n_z or n_r.')

    else:
        raise ValueError('Unsupported geometry')

    geometry.voxelization = voxelization
    #geometry.matrix_E_dep = voxelization.matrix
    return geometry

##### Geometry classes
class Ortho:
    def __init__(self, x, y, z, z_ch):
        self.x = x
        self.y = y
        self.z = z
        self.x_left = -x / 2.
        self.x_right = x / 2.
        self.y_left = -y / 2.
        self.y_right = y / 2.
        self.z_bott = 0.
        self.z_top = z

        self.cur_position = np.array([0., 0., 0.])
        self.cur_x = 0.
        self.cur_y = 0.
        self.cur_z = 0.
        self.cur_med = 0
        self.cur_dist = 0.
        
        self.z_ch = z_ch
        if z_ch is None:
            self.N_media = 1
            self.init_medium = self.nothing
        else:
            self.N_media = 2

    def nothing(self, *arg):
        pass

    def try_position(self, position):
        # position # [x, y, z] cartesian coordinates of point
        self.temp_position = position
        self.temp_x = position[0]
        self.temp_y = position[1]
        self.temp_z = position[2]

    def in_out(self):
        # Update to temp_position and check if it is in/out the medium
        self.cur_position = self.temp_position
        self.cur_x = self.temp_x
        self.cur_y = self.temp_y
        self.cur_z = self.temp_z
        if self.cur_x < self.x_left or self.cur_x > self.x_right:
            return False
        elif self.cur_y < self.y_left or self.cur_y > self.y_right:
            return False
        elif self.cur_z < self.z_bott or self.cur_z > self.z_top:
            return False
        else:
            return True

    def init_medium(self, theta, phi):
        # For a new particle after in_out
        # obtain the medium of the current position
        dist = self.cur_z - self.z_ch
        self.cur_dist = dist
        if dist<0.: # first medium
            self.cur_med = 0
        elif dist>0.: # second medium
            self.cur_med = 1
        else: # interface
            self.update_medium(theta, phi)
            
    def update_medium(self, theta, phi):
        # For a particle on the interface
        # decide the medium depending on its propagation direction
        if theta<=np.pi/2.: #upwards or horizontal
            self.cur_med = 1
        else: #downwards
            self.cur_med = 0

    def update_position(self, p_forw, s):
        # Check if the particle changes of medium
        # If so, transport the particle to the interface and set temp_position and cur_dist
        # (Note that cur_position is then updated in in_out, but not cur_dist)
        # Output True/False for change of medium, the current position
        # and the track length correction factor
        self.try_position(p_forw)
        dist = self.temp_z - self.z_ch

        if dist<0.:
            med = 0
        elif dist>0.:
            med = 1
        else:
            med = None #interface

        if med==self.cur_med: # No change of medium
            self.cur_dist = dist
            return False, p_forw, 1.

        if med is None: # particle reaching just the interface
            self.cur_dist = 0.
            return True, p_forw, 1.

        direction = p_forw - self.cur_position
        sol = abs(self.cur_dist / direction[2]) # 0<sol<1
        p_forw = self.cur_position + sol * direction
        self.try_position(p_forw)
        self.cur_dist = 0.
        return True, p_forw, sol

    def plot(self):
        x = np.linspace(self.x_left, self.x_right, 50)
        y = np.linspace(self.y_left, self.y_right, 50)
        z = np.linspace(self.z_bott, self.z_top, 50)

        fig = plt.figure()
        x_grid, y_grid = np.meshgrid(x, y)
        z1_grid = (np.ones_like(x_grid))*self.z_bott
        z2_grid = (np.ones_like(x_grid))*self.z_top
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_grid, y_grid, z1_grid, color = 'c', alpha=0.25)
        ax.plot_surface(x_grid, y_grid, z2_grid, color = 'c', alpha=0.25)

        x_grid, z_grid = np. meshgrid(x, z)
        y1_grid = (np.ones_like(x_grid))*self.y_left
        y2_grid = (np.ones_like(x_grid))*self.y_right
        ax.plot_surface(x_grid, y1_grid, z_grid, color = 'c', alpha=0.25)
        ax.plot_surface(x_grid, y2_grid, z_grid, color = 'c', alpha=0.25)

        y_grid, z_grid = np. meshgrid(y, z)
        x1_grid = (np.ones_like(x_grid))*self.x_left
        x2_grid = (np.ones_like(x_grid))*self.x_right
        ax.plot_surface(x1_grid, y_grid, z_grid, color = 'c', alpha=0.25)
        ax.plot_surface(x2_grid, y_grid, z_grid, color = 'c', alpha=0.25)

        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_zlabel("z (cm)")

        smax = max(self.x, self.y, self.z)
        ax.set_zlim(0., smax)
        ax.set_xlim(-smax/2., smax/2.)
        ax.set_ylim(-smax/2., smax/2.)

        return ax

    def Edep_init(self):
        self.voxelization.matrix = np.zeros_like(self.voxelization.matrix)

    def Edep_update(self, Edep):
        self.voxelization.update(self, Edep)

    def Edep_out(self, n_phot):
        return self.voxelization.out(n_phot)

    def Edep_save(self, E_dep, name, E_max, E_save):
        return self.voxelization.save(E_dep, name, E_max, E_save)

    def Edep_plot(self, E_dep):
        self.voxelization.plot(E_dep)

    def tracks(self, p1, p2, ax):
        track_x = [p1[0], p2[0]]
        track_y = [p1[1], p2[1]]
        track_z = [p1[2], p2[2]]
        ax.plot3D(track_x, track_y, track_z, linewidth = 0.5, color = 'blue')

    def points(self, p1, Edep, E0, ax):
        if Edep > 0.95 * E0:
            color = 'r'
        elif Edep > 0.5 * E0:
            color = 'orange'
        elif Edep > 0.1 * E0:
            color = 'g'
        elif Edep > 0.:
            color = 'y'
        else:
            color = 'k'
        ax.scatter3D(p1[0], p1[1], p1[2], c = color, s = 10.)


class Cylinder(Ortho):
    def __init__(self, r, z, z_ch, r_ch):
        super().__init__(2.*r, 2.*r, z, z_ch)
        self.r = r
        self.cur_r2 = 0.
        self.cur_r = 0.

        self.z_ch = z_ch
        self.r_ch = r_ch
        if z_ch is None and r_ch is None:
            self.N_media = 1
            self.r2_ch = None
            self.init_medium = self.nothing
        else:
            self.N_media = 2
            if z_ch is not None:
                self.r_ch = None # If z_ch is given, r_ch is ignored
                self.r2_ch = None
                # init_medium, update_medium and update_position already defined for z_ch
            else:
                self.r2_ch = r_ch**2
                # overwrite methods for r_ch
                self.init_medium = self.init_medium_r
                self.update_medium = self.update_medium_r
                self.update_position = self.update_position_r

    def try_position(self, position):
        #position # [x, y, z] cartesian coordinates of point
        self.temp_position = position
        self.temp_x = position[0]
        self.temp_y = position[1]
        self.temp_z = position[2]
        #radial distance in cylindrical coordinates
        self.temp_r2 = self.temp_x**2 + self.temp_y**2 # r^2 coordinate
        self.temp_r = self.temp_r2**0.5
    
    def in_out(self):
        # Update to temp_position and check if it is in/out the medium
        self.cur_position = self.temp_position
        self.cur_x = self.temp_x
        self.cur_y = self.temp_y
        self.cur_z = self.temp_z
        self.cur_r2 = self.temp_r2
        self.cur_r = self.temp_r
        
        if self.cur_r > self.r:
            return False
        elif self.cur_z < self.z_bott or self.cur_z > self.z_top:
            return False
        else:
            return True

    def init_medium_r(self, theta, phi):
        # For a new particle after in_out
        # obtain the medium of the current position
        dist = self.cur_r - self.r_ch
        self.cur_dist = dist
        if dist<0.: # first medium
            self.cur_med = 0
        elif dist>0.: # second medium
            self.cur_med = 1
        else: # interface
            self.update_medium(theta, phi)
            
    def update_medium_r(self, theta, phi):
        # For a particle on the interface
        # decide the medium depending on its propagation direction
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        proj = self.cur_x*cos_phi + self.cur_y*sin_phi
        if proj<0.: #inwards
            self.cur_med = 0
        else: #outwards
            self.cur_med = 1 

    def update_position_r(self, p_forw, s):
        # Check if the particle changes of medium
        # If so, transport the particle to the interface and set temp_position and cur_dist
        # (Note that cur_position is then updated in in_out, but not cur_dist)
        # Output True/False for change of medium, the current position
        # and the track length correction factor
        cur_dist = self.cur_dist
        d = abs(cur_dist)
        self.try_position(p_forw)
        dist = self.temp_r - self.r_ch
        self.cur_dist = dist

        if s<d: # Step not reaching the interface
            return False, p_forw, 1.

        if cur_dist<=0. and dist<0.: # Step inside the inner cylinder
            return False, p_forw, 1.
        if cur_dist==0. and self.cur_med==1: # Step from the interface in the outer cylinder
            return False, p_forw, 1.

        direction = p_forw - self.cur_position
        t2 = direction[0]**2 + direction[1]**2
        t = t2**0.5 # Length of step projection on XY
        if t<d: # Step not reaching the interface
            return False, p_forw, 1.

        # To transport the particle to the interface
        # p_int = cur_pos + sol * direction, 0 < sol < 1
        # sol = -b +- sqrt(rad)
        dif_r2 = self.cur_r2 - self.r2_ch # >0 from outer cylinder
        b = (self.cur_x*direction[0] + self.cur_y*direction[1]) / t2
        rad = b**2 - dif_r2/t2 # radicand
        if rad<0.: # Step inside the outer cylinder (no solution)
            return False, p_forw, 1.

        root = rad**0.5
        sol = -b - root # lower solution
        # Discard both backward steps and 0-length steps (just from the interface)
        if sol<=0. or cur_dist==0.:
            sol = -b + root # higher solution
            if sol<=0.: # Step moving away from the interface
                return False, p_forw, 1.

        if sol>1.: # Step not reaching the interface
            return False, p_forw, 1.

        # The step reaches the interface, so the current position is updated
        p_forw = self.cur_position + sol * direction
        self.try_position(p_forw) 
        self.cur_dist = 0.
        return True, p_forw, sol
        
    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        z = np.linspace(0., self.z, 50)
        theta = np.linspace(0., 2. * np.pi, 50)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = self.r * np.cos(theta_grid)
        y_grid = self.r * np.sin(theta_grid)
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.25)

        p = Circle((0, 0), self.r, color = 'c', alpha = 0.25)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")
        p = Circle((0, 0), self.r, color = 'c', alpha = 0.25)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=self.z, zdir="z")

        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_zlabel("z (cm)")

        smax = max(2.*self.r, self.z)
        ax.set_zlim(0, smax)
        ax.set_xlim(-smax/2, smax/2)
        ax.set_ylim(-smax/2, smax/2)

        return ax

class Sphere(Cylinder):
    def __init__(self, r, z_ch, r_ch):
        super().__init__(2.*r, 2.*r, 2.*r, z_ch)
        self.r = r
        self.cur_r2 = 0.
        self.cur_r = 0.
        self.z_bott = -r
        self.z_top = r

        self.z_ch = z_ch
        self.r_ch = r_ch
        if z_ch is None and r_ch is None:
            self.N_media = 1
            self.r2_ch = None
            self.init_medium = self.nothing
        else:
            self.N_media = 2
            if z_ch is not None:
                self.r_ch = None # If z_ch is given, r_ch is ignored
                self.r2_ch = None
                # init_medium, update_medium and update_position already defined for z_ch
            else:
                self.r2_ch = r_ch**2
                # overwrite methods for r_ch
                self.init_medium = self.init_medium_r
                self.update_medium = self.update_medium_r
                self.update_position = self.update_position_r

    def try_position(self, position):
        #position # [x, y, z] cartesian coordinates of point
        self.temp_position = position
        self.temp_x = position[0]
        self.temp_y = position[1]
        self.temp_z = position[2]
        #radial distance in spherical coordinates
        self.temp_r2 = self.temp_x**2 + self.temp_y**2 + self.temp_z**2
        self.temp_r = self.temp_r2**0.5

    def in_out(self):
        # Update to temp_position and check if it is in/out the medium
        self.cur_position = self.temp_position
        self.cur_x = self.temp_x
        self.cur_y = self.temp_y
        self.cur_z = self.temp_z
        self.cur_r2 = self.temp_r2
        self.cur_r = self.temp_r

        if self.cur_r > self.r:
            return False
        else:
            return True

    def update_medium_r(self, theta, phi):
        # For a particle on the interface
        # decide the medium depending on its propagation direction
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        proj = (self.cur_x*sin_theta*cos_phi + self.cur_y*sin_theta*sin_phi
                + self.cur_z*cos_theta)
        if proj<0.: #inwards
            self.cur_med = 0
        else: #outwards
            self.cur_med = 1 

    def update_position_r(self, p_forw, s):
        # Check if the particle changes of medium
        # If so, transport the particle to the interface and set temp_position and cur_dist
        # (Note that cur_position is then updated in in_out, but not cur_dist)
        # Output True/False for change of medium, the current position
        # and the track length correction factor
        cur_dist = self.cur_dist
        d = abs(cur_dist)
        self.try_position(p_forw)
        dist = self.temp_r - self.r_ch
        self.cur_dist = dist

        if s<d: # Step not reaching the interface
            return False, p_forw, 1.

        if cur_dist<=0. and dist<0.: # Step inside the inner sphere
            return False, p_forw, 1.
        if cur_dist==0. and self.cur_med==1: # Step from the interface in the outer sphere
            return False, p_forw, 1.

        direction = p_forw - self.cur_position
        s2 = s*s
        # To transport the particle to the interface
        # p_int = cur_pos + sol * direction, 0 < sol < 1
        # sol = -b +- sqrt(rad)
        dif_r2 = self.cur_r2 - self.r2_ch # >0 from outer sphere
        b = (self.cur_x*direction[0] + self.cur_y*direction[1] + self.cur_z*direction[2]) / s2
        rad = b**2 - dif_r2/s2 # radicand
        if rad<0.: # Step inside the outer sphere (no solution)
            return False, p_forw, 1.

        root = rad**0.5
        sol = -b - root # lower solution
        # Discard both backward steps and 0-length steps (just from the interface)
        if sol<=0. or cur_dist==0.:
            sol = -b + root # higher solution
            if sol<=0.: # Step moving away from the interface
                return False, p_forw, 1.

        if sol>1.: # Step not reaching the interface
            return False, p_forw, 1.

        # The step reaches the interface, so the current position is updated
        p_forw = self.cur_position + sol * direction
        self.try_position(p_forw) 
        self.cur_dist = 0.
        return True, p_forw, sol

    def plot(self):
        N=200
        u = np.linspace(0., 2. * np.pi, N)
        v = np.linspace(0., np.pi, N)
        x = np.outer(np.cos(u), np.sin(v)) * self.r
        y = np.outer(np.sin(u), np.sin(v)) * self.r
        z = np.outer(np.ones_like(u), np.cos(v)) * self.r
        stride=2

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z, linewidth=0.0, cstride=stride, rstride=stride, alpha=0.25)

        ax.set_zlim(-self.r, self.r)
        ax.set_xlim(-1.3*self.r, 1.3*self.r)
        ax.set_ylim(-1.3*self.r, 1.3*self.r)

        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_zlabel("z (cm)")
        return ax

class cart_vox:
    def __init__(self, geom, n_x, n_y, n_z):
        self.x = geom.x
        self.y = geom.y
        self.z = geom.z
        self.x_left = geom.x_left
        self.x_right = geom.x_right
        self.y_left = geom.y_left
        self.y_right = geom.y_right
        self.z_bott = geom.z_bott
        self.z_top = geom.z_top

        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        self.matrix = np.zeros((n_x, n_y, n_z))

        self.delta_x = self.x / self.n_x
        self.delta_y = self.y / self.n_y
        self.delta_z = self.z / self.n_z
        self.delta_v = self.delta_x * self.delta_y * self.delta_z

        # To calculate the fluence along the z axis
        self.delta_r2 = min(self.delta_x/2., self.delta_y/2.)**2

    def update(self, geom, E_dep): # add energy deposited in the voxel x,y,z
        if geom.cur_x==self.x_right:
            ix = self.n_x - 1
        else:
            ix = int((geom.cur_x - self.x_left) / self.delta_x)  # voxel x-index for deposited energy matrix
        if geom.cur_y==self.y_right:
            iy = self.n_y - 1
        else:
            iy = int((geom.cur_y - self.y_left) / self.delta_y)  # voxel y-index for deposited energy matrix
        if geom.cur_z==self.z_top:
            iz = self.n_z - 1
        else:
            iz = int((geom.cur_z -self.z_bott) / self.delta_z)   # voxel z-index for deposited energy matrix

        self.matrix[ix, iy, iz] += E_dep

    def out(self, n_phot): # return E_dep matrix normalized (per unit volume)
        return self.matrix / self.delta_v * 1000. / n_phot

    def save(self, E_dep, name, E_max, E_save):
        if E_save:
            m_type = name
            En = "{:.{}f}".format( E_max, 2 ) + 'MeV'
            f_name = 'Edep_'+ m_type + '_' + En + '.npy' 
            open(f_name, "w")
            np.save(f_name, E_dep)
        return E_dep

    def plot(self, E_dep):
        extent = (self.x_left, self.x_right, self.y_left, self.y_right)
        vmax = np.log10(np.amax(E_dep))
        n_plots = min(4, self.n_z)
        fig_width = min(15., n_plots*15./4.)
        scale_factor = fig_width * self.y / self.x / n_plots
        fig, ax = plt.subplots(1, n_plots, figsize=(fig_width, scale_factor), constrained_layout=True)
        if n_plots==1:
            # log(E_dep) of iz layer [:,:,iz], transposed and rotated along the x-axis [::-1] to match the image xy axes
            E_dep_log_z = np.log10(np.transpose(E_dep[:,:,0])[::-1])
            psm = ax.imshow(E_dep_log_z, vmax = vmax, extent=extent)
            ax.set_xlabel('x (cm)')
            ax.set_ylabel('y (cm)')

        else:
            for im in range(n_plots):
                iz = (im*(self.n_z-1)) // (n_plots-1)
                # log(E_dep) of iz layer [:,:,iz], transposed and rotated along the x-axis [::-1] to match the image xy axes
                E_dep_log_z = np.log10(np.transpose(E_dep[:,:,iz])[::-1])
                psm = ax[im].imshow(E_dep_log_z, vmax = vmax, extent=extent)
                z_plot = self.z_bott + (iz + 0.5) * self.delta_z
                ax[im].set_title('z = ' + str(z_plot) +' cm')
                ax[im].set_xlabel('x (cm)')
                ax[im].set_ylabel('y (cm)')

        # Color bar attached to last plot
        cbar = fig.colorbar(psm, aspect = 50., shrink = 0.7)
        cbar.ax.set_ylabel('log(E$_{dep}$ (keV cm$^{-3}$))')

    def min_delta(self):
        return min(self.delta_x,self.delta_y,self.delta_z)*10000.

class cyl_vox:
    def __init__(self, geom, n_z, n_r):
        self.z = geom.z
        self.r = geom.r
        self.z_bott = geom.z_bott
        self.z_top = geom.z_top
        self.n_z = n_z
        self.n_r = n_r
        self.matrix = np.zeros((n_z, n_r))

        self.delta_z = geom.z / self.n_z
        self.delta_r = geom.r / self.n_r
        r2 = np.arange(self.n_r+1)**2 * self.delta_r * self.delta_r
        self.delta_v = self.delta_z * np.pi * (r2[1:]-r2[:-1])

        self.rbin = np.arange(self.delta_r/2., self.r, self.delta_r) # array of central position of bins
        self.zbin = np.arange(self.delta_z/2., self.z, self.delta_z) # array of central position of bins

        self.delta_r2 = self.delta_r**2

    def update(self, geom, E_dep): # add energy deposited in the voxel r,z
        if geom.cur_z==self.z_top:
            iz = self.n_z - 1
        else:
            iz = int((geom.cur_z - self.z_bott) / self.delta_z)  # voxel z-index for deposited energy matrix
        if geom.cur_r==self.r:
            ir = self.n_r - 1
        else:
            ir = int(geom.cur_r / self.delta_r)  # voxel r-index for deposited energy matrix

        self.matrix[iz, ir] += E_dep

    def out(self, n_phot): # return E_dep matrix normalized (keV / cm^3 / photon)
        matrix_norm = self.matrix.copy()
        k = 1000. / n_phot / self.delta_v
        for z in range(self.n_z):
            matrix_norm[z] = matrix_norm[z] * k
        # matrix_norm = np.ndarray.transpose(matrix_E_dep_norm) # z - column; r - row
        return matrix_norm

    def save(self, E_dep, name, E_max, E_save):
        # Save pandas dataframe to excel
        E_dep_df = pd.DataFrame(E_dep, columns = self.rbin, index = self.zbin)
        E_dep_df.index.name = 'z(cm)'
        E_dep_df.columns.name = 'r(cm)'
        if E_save:
            En = "{:.{}f}".format( E_max, 2 ) + 'MeV'
            m_type = name
            exc_name = 'Edep_'+ m_type + '_' + En + '.xlsx'

            open(exc_name, "w") # to excel file
            E_dep_df.to_excel(exc_name, sheet_name = 'Edep(r,z)', header = 'r(cm)', float_format = '%.3e') # includes bining data
            print(exc_name + ' written onto disk.; columns = r(cm), rows = z(cm)')
            print()
        return E_dep_df

    def plot(self, E_dep):
        # figure for color plot 2D
        # fig_2D, ax_2D = plt.subplots(figsize=(5, 5), constrained_layout=True)
        Edep = E_dep.copy()
        fig_2D, ax_2D = plt.subplots(constrained_layout=True)

        extent = (0., self.r, self.z_bott, self.z_top)
        vmax = np.log10(np.amax(E_dep))
        psm = ax_2D.imshow(np.log10(E_dep)[::-1], vmax = vmax, extent = extent, aspect = 'auto')
        ax_2D.set_xlabel('R (cm)')
        ax_2D.set_ylabel('Depth (cm)')
        ax_2D.set_title('Energy deposition')
        cbar = fig_2D.colorbar(psm, aspect = 50., shrink = 0.85, label = 'log(E$_{dep}$ (keV cm$^{-3}$))')

        #if np.count_nonzero(E_dep) == E_dep.size:
        Edep_min = None
        if np.count_nonzero(Edep)<Edep.size:
            print('Not enough data to fill the histograms for projections of the E_dep matrix.')
            Edep_min = E_dep[E_dep>0.].min()/10.
            Edep[Edep==0.] = Edep_min #To fix log errors

        # figure for depth and radial projections
        fig, ax = plt.subplots(ncols = 2, figsize=(10, 7), constrained_layout=True)
        for iz in range(0, self.n_z, 1 + self.n_z // 4):
            E = Edep[iz,:]
            color = (rand.random(), rand.random(), rand.random(), 0.40)
            z_round = str(round(iz*self.delta_z + self.delta_z/2., 1)) # round z to one decimal place
            label = z_round + ' cm'
            # print(E)
            ax[0].bar(self.rbin, E, self.delta_r, color=color, fill = True, label = label)
            ax[0].set_xlabel('R (cm)')
            ax[0].set_ylabel('E$_{dep}$ (keV cm$^{-3}$)')
            ax[0].set_title('Energy deposition vs. radial distance'+'\n'+ 'for several depths')
            ax[0].set_yscale('log')
            ax[0].set_ylim(ymin=Edep_min)
            ax[0].legend()

        # plot of radial projection
        for ir in range(0, self.n_r, 1 + self.n_r // 4):
            E = Edep[:,ir]
            color = (rand.random(), rand.random(), rand.random(), 0.40)
            r_round = str(round((ir + 0.5) * self.delta_r, 1))
            label = r_round + ' cm'
            ax[1].bar(self.zbin, E, self.delta_z, color=color, fill = True, label = label)
            ax[1].set_xlabel('Depth (cm)')
            # ax_pr[1].set_ylabel('E$_{dep}$ (keV cm$^{-3}$)')
            ax[1].set_title('Energy deposition in depth'+'\n'+ 'for several radial distances')
            ax[1].set_yscale('log')
            ax[1].set_ylim(ymin=Edep_min)
            ax[1].legend()
        #else:
        #    print('Not enough data to show projections of the E_dep matrix.')

    def min_delta(self):
        return min(self.delta_r,self.delta_z)*10000.

class sph_vox:
    def __init__(self, geom, n_r):
        self.n_r = n_r
        self.matrix = np.zeros(n_r)

        self.delta_r = geom.r / self.n_r
        r3 = np.arange(self.n_r+1)**3 * self.delta_r**3
        self.delta_v = 4./3. * np.pi * (r3[1:]-r3[:-1])

        self.rbin = np.arange(self.delta_r/2., geom.r, self.delta_r)

        self.delta_r2 = self.delta_r**2

    def update(self, geom, E_dep): # add energy deposited in the voxel r
        if geom.cur_r==geom.r:
            ir = self.n_r - 1
        else:
            ir = int(geom.cur_r * self.n_r / geom.r)  # voxel r-index for deposited energy matrix

        self.matrix[ir] += E_dep

    def out(self, n_phot): # return E_dep matrix normalized (per unit volume)
        return self.matrix / self.delta_v * 1000. / n_phot

    def save(self, E_dep, name, E_max, E_save):
        # Save pandas dataframe to excel
        E_dep_df = pd.DataFrame(E_dep, index = self.rbin, columns = ['keV/cm^3'])
        E_dep_df.index.name = 'R(cm)'
        if E_save:
            En = "{:.{}f}".format( E_max, 2 ) + 'MeV'
            m_type = name
            exc_name = 'Edep_'+ m_type + '_' + En + '.xlsx'    

            open(exc_name, "w") # to excel file
            E_dep_df.to_excel(exc_name, sheet_name = 'Edep(R)', float_format = '%.3e') # includes bining data
            print(exc_name + ' written onto disk')
            print()
        return E_dep_df

    def plot(self, E_dep):
        # figure for radial dependence
        fig, ax = plt.subplots(constrained_layout=True)
        ax.bar(self.rbin, E_dep, self.delta_r, fill = True)
        ax.set_xlabel('R (cm)')
        ax.set_ylabel('E$_{dep}$ (keV cm$^{-3}$)')
        ax.set_title('Energy deposition vs. radial distance')
        ax.set_yscale('log')

    def min_delta(self):
        return self.delta_r*10000.
