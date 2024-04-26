import random
import math

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