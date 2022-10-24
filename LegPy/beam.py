# Author: Fernando Arqueros

import numpy as np
import random
import math

#JR: Añado tipo de partícula particle = 'gamma' or 'electron'
def Beam(name = 'parallel', diam = 0.0, length = 0., r_width = 0., x_ap = 0., y_ap = 0., theta = 0.0, phi = 0.0,
         p_in = np.array ([0., 0., 0.]), particle = 'gamma'):

    theta, phi = math.radians(theta), math.radians(phi)
    
    if name == 'parallel':
        if theta >= math.pi/2.:
            raise ValueError('Please, input theta < 90 deg.')
        if diam==0.:
            beam = Pencil(theta, phi, p_in, particle)
        else:      
            beam = Broad(theta, phi, p_in, diam, particle)

    elif name == 'isotropic':
        if diam==0.: 
            if x_ap==0. and y_ap==0.:
                beam = Isotropic_in(p_in, length, r_width, particle)
        
            else:
                if p_in[2]<0.:
                    dist = abs(p_in[2])
                    xy_in = p_in[0:-1] #xy coordinates of the source/focus
                    xy_ap = np.array([[x_ap, y_ap],[-x_ap, y_ap],[x_ap, -y_ap],[-x_ap, -y_ap]]) # aperture size
                    XY = xy_in + xy_ap # coordinates of the aperture corners
                    diam_e = 2 * max(np.linalg.norm(XY[0]), np.linalg.norm(XY[1]),np.linalg.norm(XY[2]),np.linalg.norm(XY[3]))
                    cos_theta_max = 1. / (1. + (diam_e/2./dist)**2)**0.5
                    beam = Isotropic_out_r(dist, p_in, x_ap, y_ap, cos_theta_max, particle)
                else:
                    raise ValueError('Please, input valid source location')      
                           
        else:
            if x_ap!=0. or y_ap!=0.: 
                raise ValueError('Please, input valid aperture type')          
            if p_in[2]<0.:            
                dist = abs(p_in[2])
                diam_e = diam + 2.* math.sqrt(p_in[0]**2 + p_in[1]**2)
                cos_theta_max = 1. / (1. + (diam_e/2./dist)**2)**0.5
                beam = Isotropic_out_c(dist, diam, p_in, cos_theta_max, particle)
            else:
                raise ValueError('Please, input valid source location')      
                    
    return beam              

#JR: Añado atributo particle en todos los tipos de beam
class Pencil:
    def __init__(self, theta, phi, p_in, particle):
        self.theta = theta
        self.phi = phi
        self.p_in = np.array(p_in)
        self.particle = particle
    def in_track(self):
        return self.p_in, self.theta, self.phi
        
class Broad:
    def __init__(self, theta, phi, p_in, diam, particle):
        self.theta = theta
        self.phi = phi
        self.p_in = np.array(p_in)
        self.diam = diam
        self.particle = particle
    def in_track(self):
        p_in_b = np.array([0.,0.,0.])
        r_cir = self.diam / 2. * math.sqrt(random.random())
        alpha = 2. * math.pi * random.random()

        p_in_b[0] = (r_cir / math.cos (self.theta) * math.sin (alpha) * math.cos (self.phi)
                     - r_cir * math.cos (alpha) * math.sin (self.phi) + self.p_in[0])
        p_in_b[1] = (r_cir / math.cos (self.theta) * math.sin (alpha) * math.sin (self.phi)
                     + r_cir * math.cos (alpha) * math.cos (self.phi) + self.p_in[1])
        p_in_b[2] = self.p_in[2]
        return p_in_b, self.theta, self.phi    

class Isotropic_out_c:
    def __init__(self, dist, diam, p_in, cos_theta_max, particle):
        self.dist = dist
        self.diam = diam
        self.p_in = np.array(p_in)        
        self.cos_theta_max = cos_theta_max
        self.particle = particle

    def in_track(self):    
        p_in_i = np.array([0.,0.,0.])
        i=0
        while i==0:
            phi = 2. * math.pi * random.random()
            theta = math.acos(1. - random.random() * (1. - self.cos_theta_max))
            p_in_i[0] = self.p_in[0] + self.dist * math.tan(theta) * math.cos(phi)
            p_in_i[1] = self.p_in[1] + self.dist * math.tan(theta) * math.sin(phi)
            if (p_in_i[0]**2 + p_in_i[1]**2) < (self.diam/2.)**2:
                i=1
                return p_in_i, theta, phi
            
class Isotropic_out_r:
    def __init__(self, dist, p_in, x_ap, y_ap, cos_theta_max, particle):
        self.dist = dist
        self.x_ap = x_ap
        self.y_ap = y_ap
        self.cos_theta_max = cos_theta_max
        self.p_in = np.array(p_in)
        self.particle = particle

    def in_track(self):    
        p_in_i = np.array([0.,0.,0.])
        i=0
        while i==0:
            phi = 2. * math.pi * random.random()
            theta = math.acos(1. - random.random() * (1. - self.cos_theta_max))
            p_in_i[0] = self.p_in[0] + self.dist * math.tan(theta) * math.cos(phi)
            p_in_i[1] = self.p_in[1] + self.dist * math.tan(theta) * math.sin(phi)
            if abs(p_in_i[0]) < self.x_ap / 2. and abs(p_in_i[1]) < self.y_ap / 2.:
                i=1
                return p_in_i, theta, phi

class Isotropic_in:
    def __init__(self, p_in, length, r_width, particle):
        self.length = length
        self.r_width = r_width
        self.p_in = np.array(p_in)
        self.particle = particle

    def in_track(self):
        delta_z = (random.random() - 0.5) * self.length
        delta_x = (random.random() - 0.5) * 2 * self.r_width
        delta_y = (random.random() - 0.5) * 2 * self.r_width
        p_in_f = self.p_in + np.array([delta_x, delta_y, delta_z])
        theta = math.acos(1. - random.random() * 2.)
        phi = 2. * math.pi * random.random() 
        return p_in_f, theta, phi
