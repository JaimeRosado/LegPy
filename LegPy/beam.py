import numpy as np
import random
import math

#JR: Añado tipo de partícula particle = 'gamma' or 'electron'
def Beam(name = 'parallel', diam = 0.0, x_s = 0., y_s = 0., z_s = 0., r_s = 0., x_ap = 0., y_ap = 0., theta = 0.0, phi = 0.0,
         p_in = np.array ([0., 0., 0.]), particle = 'gamma'):

    theta, phi = math.radians(theta), math.radians(phi)
    
    if name == 'parallel':
        if theta >= math.pi/2.:
            raise ValueError('Please, input theta < 90 deg.')
        if diam==0.:
            beam = Pencil(theta, phi, p_in, particle)
        else: # if 'parallel' diam is the beam diameter     
            beam = Broad(theta, phi, p_in, diam, particle)

    elif name == 'isotropic':
        # if 'isotropic' diam is the diameter of a circular aperture of an external diverget beam
        if diam==0.: # 'isotropic' without circular aperture (diam) 
            if x_ap==0. and y_ap==0.: # without rectangular aperture 
                # => Source inside the medium
                    
                if x_s * y_s != 0. or x_s * z_s != 0. or y_s * z_s != 0.: # Source is an orthohedron centered at p_in.
                    # One of the dimensions can be null (flat rectangle)
                    beam = Isotropic_in_ort(p_in, x_s, y_s, z_s, particle)
                    if r_s != 0.:
                        print('Warning: Source is an Orthohedron. Source radius is ignored') 
                         
                elif z_s != 0.: # Cylindrical source (z_s, r_s). r_s might be equals 0 (segment along z direction)
                    beam = Isotropic_in_cyl(p_in, z_s, r_s, particle)
                    if x_s != 0. or y_s != 0.:
                        print('Warning: Cylindrical source. x, y dimensions are ignored') 

                elif r_s != 0.: # Source is a sphere
                    beam = Isotropic_in_sph(p_in, r_s, particle)
                    if x_s != 0. or y_s != 0.:
                        print('Warning: Spherical source. x, y dimensions are ignored')
                        
                elif x_s == 0. and y_s == 0. and z_s == 0. and r_s == 0.: # Point source
                    beam = Isotropic_in_p(p_in, particle)
                    
                else:
                    raise ValueError('Please, input correct source size parameters')
              
            else: # source outside the medium (or divergent beam) with rectangular aperture
                if p_in[2]<0.:
                    dist = abs(p_in[2])
                    xy_in = p_in[0:-1] #xy coordinates of the source/focus might be != (0,0)
                    xy_ap = np.array([[x_ap, y_ap],[-x_ap, y_ap],[x_ap, -y_ap],[-x_ap, -y_ap]]) # aperture size
                    XY = xy_in + xy_ap # coordinates of the aperture corners
                    diam_e = 2 * max(np.linalg.norm(XY[0]), np.linalg.norm(XY[1]),np.linalg.norm(XY[2]),np.linalg.norm(XY[3]))
                    cos_theta_max = 1. / (1. + (diam_e/2./dist)**2)**0.5
                    beam = Isotropic_out_r(dist, p_in, x_ap, y_ap, cos_theta_max, particle)
                else:
                    raise ValueError('Please, input valid source location outside the medium')      
                           
        else: # source outside the medium (or divergent beam) with circular aperture  
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

class Isotropic_in_p: # point source
    def __init__(self, p_in, particle):
        self.p_in = np.array(p_in)
        self.particle = particle
        
    def in_track(self):
        theta = math.acos(1. - random.random() * 2.)
        phi = 2. * math.pi * random.random() 
        p_in = self.p_in
        return p_in, theta, phi

class Isotropic_in_ort: # orthohedro
    def __init__(self, p_in, x_s, y_s, z_s, particle):
        self.p_in = np.array(p_in)
        self.particle = particle
        self.x_s = x_s
        self.y_s = y_s
        self.z_s = z_s
        
    def in_track(self):
        delta_x = (random.random() - 0.5) * self.x_s
        delta_y = (random.random() - 0.5) * self.y_s
        delta_z = (random.random() - 0.5) * self.z_s
        p_in_f = self.p_in + np.array([delta_x, delta_y, delta_z])
        theta = math.acos(1. - random.random() * 2.)
        phi = 2. * math.pi * random.random() 
        return p_in_f, theta, phi

class Isotropic_in_cyl: # cylindrical
    def __init__(self, p_in, z_s, r_s, particle):
        self.p_in = np.array(p_in)
        self.particle = particle
        self.r_s = r_s
        self.z_s = z_s
        
    def in_track(self):
        delta_z = (random.random() - 0.5) * self.z_s
        i=0
        while i==0:
            delta_x = (random.random() - 0.5) * 2 * self.r_s
            delta_y = (random.random() - 0.5) * 2 * self.r_s
            if delta_x**2 + delta_y**2 <= self.r_s * self.r_s:
                i = 1
            p_in_f = self.p_in + np.array([delta_x, delta_y, delta_z])
        theta = math.acos(1. - random.random() * 2.)
        phi = 2. * math.pi * random.random() 
        return p_in_f, theta, phi

class Isotropic_in_sph: # sphere
    def __init__(self, p_in, r_s, particle):
        self.p_in = np.array(p_in)
        self.particle = particle
        self.r_s = r_s
        
    def in_track(self): 
        i=0
        while i==0:
            delta_z = (random.random() - 0.5) * 2 * self.r_s
            delta_x = (random.random() - 0.5) * 2 * self.r_s
            delta_y = (random.random() - 0.5) * 2 * self.r_s
            if delta_x**2 + delta_y**2 + delta_z**2 <= self.r_s * self.r_s:
                i = 1
            p_in_f = self.p_in + np.array([delta_x, delta_y, delta_z])
        theta = math.acos(1. - random.random() * 2.)
        phi = 2. * math.pi * random.random() 
        return p_in_f, theta, phi