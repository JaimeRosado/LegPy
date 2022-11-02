# Author: Fernando Arqueros

import random
import math
# import numpy
#import matplotlib.pyplot as plt

#JR: Pongo gamma como parámetro de entrada
def theta_KN(gamma):
    while True:
        rth = random.random() * math.pi
        cos_rth = math.cos(rth)
        sin_rth = math.sin(rth)
        rKNc = random.random() * 2.75e-25
        a1 = sin_rth / (1.+gamma*(1.-cos_rth))**2
        a2 = (gamma*gamma*(1.-cos_rth)**2) / (1.+gamma*(1.-cos_rth))
        rKN = 2.494672e-25 * a1 * (1+cos_rth**2+a2)
        if rKN > rKNc:
            return rth

#JR: Paso la función a este módulo y simplifico para que sólo calcule el ángulo que se desvía el electrón
#Uso librería math
def Compton_electron(gamma, theta_C):
    rth = math.atan(1. / ((1.+gamma)*math.tan(theta_C/2.)))
    return rth
        
def phi_ang():
    rph = random.random() * 2. * math.pi
    return rph

def theta_isotropic():
    return math.acos(2. * random.random() - 1.)

def theta_Ray_Sc():
    while True:
        rth = random.random()*math.pi
        extr = (2./3.)**(0.5) # of the normalized Thomson scattering law
        rRsc = random.random() * extr
        Rsc = 3./8. * math.sin(rth)*(1+math.cos(rth)**2) # 3/8 normalitation constant
        if Rsc > rRsc:
            return rth

def theta_phi_new_frame(theta_1, phi_1, theta, phi):
    if theta_1==0.:
        return (theta, phi)
    sin_theta_1 = round(math.sin(theta_1), 5)
    if sin_theta_1==0.:
        return (math.pi-theta, phi)
    cos_theta_2 = math.cos(theta_1) * math.cos(theta) + math.sin(theta_1) * math.sin(theta) * math.cos(phi)
    theta_2 = math.acos(cos_theta_2)
    sin_theta_2 = math.sin(theta_2)
    cos_phi2_phi1 = (math.cos(theta) - cos_theta_2 * math.cos(theta_1)) / sin_theta_1 / sin_theta_2
    cos_phi2_phi1 = min(1., cos_phi2_phi1)
    cos_phi2_phi1 = max(-1., cos_phi2_phi1)

    sin_phi2_phi1 = math.sin(theta) * math.sin(phi) / sin_theta_2
    acos_ph2_ph1 = math.acos(cos_phi2_phi1)

    if sin_phi2_phi1 > 0.:
        phi_2 = acos_ph2_ph1 + phi_1
    else:
        phi_2 = 2. * math.pi - acos_ph2_ph1 + phi_1  
    phi_2 = phi_2 % (2. * math.pi)
    return (theta_2, phi_2)

def rotate(vx, vy, vz, rot_angle, x, y, z):
    """
    Rotate the vector (x,y,z) by an angle rot_angle (positive or negative)
    around the axis (vx,vy,vz), where v is a unit vector.
    """
    ct = math.cos(rot_angle)
    st = math.sin(rot_angle)
    x_rot = ((ct+vx*vx*(1.-ct)) * x + (vx*vy*(1.-ct)-vz*st) * y
             + (vx*vz*(1.-ct)+vy*st) * z)
    y_rot = ((vx*vy*(1.-ct)+vz*st) * x + (ct+vy*vy*(1.-ct)) * y
             + (vy*vz*(1.-ct)-vx*st) * z)
    z_rot = ((vx*vz*(1.-ct)-vy*st) * x + (vy*vz*(1.-ct)+vx*st) * y
             + (ct+vz*vz*(1.-ct)) * z)

    return (x_rot, y_rot, z_rot)


# theta_1 = 0.01 # deg.
# phi_1 = 45.0 # deg.

# theta_1 = math.radians(theta_1) # 
# phi_1 = math.radians(phi_1) # 
# #
# x_1 = math.sin(theta_1)*math.cos(phi_1)
# y_1 = math.sin(theta_1)*math.sin(phi_1)
# z_1 = math.cos(theta_1)

# x_rot = math.sin(phi_1)
# y_rot = -math.cos(phi_1)
# z_rot = 0.
# # print(x_1, y_1, z_1)
# # print(x_rot, y_rot, z_rot)
# theta_rot = -math.acos(z_1)
# # print(theta_rot)

# theta_C = math.radians(60.0) # deg.
# phi_C = math.radians(40.0) # deg.
# #
# x_C = math.sin(theta_C)*math.cos(phi_C)
# y_C = math.sin(theta_C)*math.sin(phi_C)
# z_C = math.cos(theta_C)
# #
# x_2, y_2, z_2 = rotate(x_rot, y_rot, z_rot, theta_rot, x_C, y_C, z_C)
# #
# theta_2 = math.acos(z_2)
# phi_2 = math.atan(y_2/z_2)
# #
# # print(math.degrees(theta_2), math.degrees(phi_2))
# #
# NA = theta_phi_new_frame(theta_1, phi_1, theta_C, phi_C)



# v_x = ([0, x_1])
# v_y = ([0, y_1])
# v_z = ([0, z_1])

# v_x_C =([0, x_C])
# v_y_C =([0, y_C])
# v_z_C =([0, z_C])

# v_x_2 = ([0, x_2])
# v_y_2 = ([0, y_2])
# v_z_2 = ([0, z_2])

# fig_test = plt.figure()
# ax_g = fig_test.add_subplot(111, projection='3d')
# ax_g.set_xlabel("x")
# ax_g.set_ylabel("y")
# ax_g.set_zlabel("z")
# ax_g.set_xlim(-1.,1.)
# ax_g.set_ylim(-1.,1.)
# ax_g.set_zlim(-1.,1.)
# ax_g.plot3D(v_x, v_y, v_z, linewidth = 1.5, color = 'blue')
# ax_g.plot3D(v_x_C, v_y_C, v_z_C, linewidth = 1.5, color = 'red')
# ax_g.plot3D(v_x_2, v_y_2, v_z_2, linewidth = 1.5, color = 'black')
# theta_2 = NA[0]
# phi_2 = NA[1]
# print(math.degrees(theta_2), math.degrees(phi_2))
# v_x_2 = ([0, math.sin(theta_2)*math.cos(phi_2)])
# v_y_2 = ([0, math.sin(theta_2)*math.sin(phi_2)])
# v_z_2 = ([0, math.cos(theta_2)])
# ax_g.plot3D(v_x_2, v_y_2, v_z_2, linewidth = 1.5, color = 'green')


# plt.show()
