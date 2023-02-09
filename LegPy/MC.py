# Author: Fernando Arqueros

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#
from .angles import theta_KN, Compton_electron, theta_isotropic, phi_ang, theta_Ray_Sc, theta_phi_new_frame
from .figures import gamma_hists, e_hists, esc_gammas, fluence_data

def Plot_beam(media, geometry, spectrum, beam):
    MC(media, geometry, spectrum, beam, n_part=50, tracks=True)

##### MC options
def MC(media, geometry, spectrum, beam, n_part=100, E_cut=0.01, tracks=False,
       points=False, Edep_matrix=False, E_save=True, E_plot=True, ang_E_gamma_out=False, 
       fluence=False, histograms=False, h_save=True, h_plot = True, n_ang=20, n_E=20, 
       n_z=20, e_transport=False, e_length=None, e_K=None, e_f=None, e_g=0.):

    # Initial particle energies for which calculations are speeded up
    if spectrum.name=="mono":
        energies = np.array([spectrum.E])
    elif spectrum.name=="multi_mono":
        energies = np.sort(spectrum.energies)
    else:
        energies = None

    if isinstance(media, list):
        N_media = len(media)
    else:
        media = [media]
        N_media = 1
    if geometry.N_media==1 and N_media>1:
        print('The input geometry does not support several media. Only the first medium is used.')
        media = [media[0]]
        N_media = 1
    elif geometry.N_media<N_media:
        N_media = geometry.N_media
        print('The input geometry only accept {} media.'.format(N_media), 'Only the first {} media are used.'.format(N_media))
        media = media[0:(N_media-1)]

    part_type = beam.particle
    if part_type=='electron' or e_transport:
        for index, medium in enumerate(media):
            e_data = medium.e_data
            if e_data is None:
                raise ValueError('No electron data is available for the medium {}'.format(index+1))
            if e_data.E_max<spectrum.E_max:
                raise ValueError('Maximum electron energy out of range of the electron data loaded to the medium {}'.format(index+1))
    
            # By default, e_length is used (instead of e_K) and set to the minimum voxel size
            vox_length = geometry.voxelization.min_delta()
            if e_length is None and e_K is None:
                e_length = vox_length
        
            # The step list is generated according to the simulation options
            e_data.make_step_list(E_cut, e_length, e_K, e_f, e_g, energies)
            # If not used the default e_length value
            if e_data.s.max()*9999.>vox_length: # To avoid some round issues in e_data.s
                print('The step length of some electrons may be greater than the voxel size.')

    if part_type!='electron':
        if spectrum.E_max>1.022: # Pair production is included
            pair = True
        else:
            pair = False

        for index, medium in enumerate(media):
            ph_data = medium.ph_data
            if ph_data is None:
                raise ValueError('No photon data is available for the medium {}'.format(index+1))
            if ph_data.E_max is not None: # NIST medium
                if ph_data.E_max<spectrum.E_max:
                    raise ValueError(
                        'Maximum photon energy out of range of the cross section data loaded to the medium {}'.format(index+1))
            ph_data.init_MC(energies, pair)

    geometry.Edep_init()
    E_max = spectrum.E_max # to skip problems in the upper limit of the histogram
    
    if N_media==1:
        def func(p_back, theta, phi, step_length):
            p_forw = take_step(p_back, theta, phi, step_length)
            geometry.try_position(p_forw)
            return False, p_forw, 1.
    else: # >1 media
        def func(p_back, theta, phi, step_length):
            p_forw = take_step(p_back, theta, phi, step_length)
            change, p_forw, k = geometry.update_position(p_forw, step_length)
            return change, p_forw, k
    if tracks:
        ax = geometry.plot()
        def part_step(p_back, theta, phi, step_length):
            change, p_forw, k = func(p_back, theta, phi, step_length)
            geometry.tracks(p_back, p_forw, ax)
            return change, p_forw, k
    else:
        part_step = func

    # These functions do nothing by default
    hists_out = nothing
    add_point = nothing
    ang_E_gamma_out_plot = nothing
    flow = nothing
    fluence_out = nothing

    # The function particle is defined according to the primary particle
    if part_type=='electron':
        def particle(position, theta, phi, E):
            return electron(media, geometry, position, theta, phi, E, part_step)

        # The function particle is modified to include histogram update
        if histograms:
            z_top = geometry.z_top
            hists = e_hists(n_z, n_ang, z_top, n_part)
            particle = add_hist_to_part(particle, hists)
            hists_out = hists.out

    else:
        # Make available the add_point function
        if points:
            if not tracks:
                ax = geometry.plot()
            def add_point(position, Edep):
                geometry.points(position, Edep, E_max, ax)

        if fluence:
            f_data = fluence_data(geometry, n_z, n_E, E_max)
            def flow(p_back, p_forw, step_length, E):
                f_data.add_count(p_back, p_forw, step_length, E)
            fluence_out = f_data.out

        def particle(position, theta, phi, E):
            output = photon(media, geometry, position, theta, phi, E, E_cut, part_step, add_point, flow, e_transport)
            return output

        # The function particle is modified to include histogram update
        if histograms:
            hists = gamma_hists(ph_data, n_ang, n_E, E_max, n_part)
            particle = add_hist_to_part(particle, hists)
            hists_out = hists.out

        # The function particle is modified to include ang_E_gamma histogram
        if ang_E_gamma_out:
            ang_E_gamma_out = esc_gammas(E_max)
            particle = add_hist_to_part(particle, ang_E_gamma_out)
            ang_E_gamma_out_plot = ang_E_gamma_out.out
           
    time_i = timeit.default_timer() # Computing time (Start)

    # loop over cases
    for part in range(n_part):
        # Initial position and direction
        # The initial position becomes the back point in the first step
        position, theta, phi = beam.in_track()
        geometry.try_position(position)
        part_in = geometry.in_out()
        if not part_in:
            raise ValueError('Some particles do not reach the medium. Check your beam geometry and try again.')
        geometry.init_medium(theta, phi)
        E = spectrum.in_energy() # initial energy
        particle(position, theta, phi, E)

    if tracks == False and points == False:
        time_f = timeit.default_timer() # Computing time (End)
        time_per_part = (time_f - time_i) / (n_part)
        print()
        print('The simulation has ended')
        print()
        print('Computing time per beam particle = ',"{:.2e}".format(time_per_part), 'seconds')
        print()

    Edep_df = None # Default value if Edep_matrix = False
    if Edep_matrix: # plot of deposited energy distribution.
        Edep = geometry.Edep_out(n_part)
        if E_save:
            name = ''
            for index, m in enumerate(media):
                name = name + m.name
                if index<N_media-1:
                    name = name + '_'
            Edep_df = geometry.Edep_save(Edep, name, E_max, E_save)
        if E_plot: # plot Edep spatial distribution
            geometry.Edep_plot(Edep)
    hists = hists_out(h_save, h_plot) 
    ang_E_gamma_out_plot()
    flu = fluence_out(n_part, h_save, h_plot)
    plt.show()
    return hists, Edep_df, flu

########   Particle simulation

### Auxiliary functions
def take_step(p_back, theta, phi, step_length):
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    # update particle direction
    Direction = np.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]) #unit vector
    return p_back + step_length * Direction # step

def nothing(*arg):
    pass

def add_hist_to_part(part, hist):
    def func(*arg):
        output = part(*arg)
        hist.add_count(output)
        return output
    return func

## photons
def photon(media, geometry, p_back, theta, phi, E, E_cut=0.01,
           step=take_step, add_point=nothing, flow=nothing, e_transport=False):
    medium = media[geometry.cur_med]
    ph_data = medium.ph_data
    E_ab = 0. # initialize absorbed energy for the new photon
    phot_in = True
    new_phot = True
    while phot_in:
        if E <= E_cut:
            geometry.Edep_update(E) # update E_dep matrix
            E_ab += E # add energy of photon
            return False, E_ab, E, theta
            
        # track length for photon energy
        step_length = ph_data.Rand_track(E)
        change, p_forw, k = step(p_back, theta, phi, step_length)
        flow(p_back, p_forw, k*step_length, E)

        phot_in = geometry.in_out()
        if not phot_in: # of the medium
            # The photon escapes without interacting
            # E_ab is set to -1 to know that the E_ab histogram should not been updated
            if new_phot:
                return False, -1, E, theta
            # The photon escapes after interacting
            else:
                return True, E_ab, E, theta

        if change:
            # The photon was transported to the intertaface between two media without interaction
            # The propagation direction (theta, phi) is mantained, so the medium changes
            # geometry.update_medium(theta, phi)
            geometry.cur_med = 1 - geometry.cur_med
            medium = media[geometry.cur_med]
            ph_data = medium.ph_data # Change medium
            p_back = p_forw
            continue

        new_phot = False # The photon interacts
        Proc = ph_data.Rand_proc(E)
        if Proc == 'Photoelectric':
            E_ab += E # add photoelectric absorption.
            if e_transport:
                electron(media, geometry, p_forw, theta, phi, E, step)
            else:
                geometry.Edep_update(E)
            add_point(p_forw, E)
            return False, E_ab, E, theta

        elif Proc == 'Compton': # Compton interaction
            # calculation of photon-Compton angles
            gamma = E / 0.511 # electron mass
            theta_C = theta_KN(gamma)
            phi_C = phi_ang()
            E_C = E / (1. + gamma * (1. - math.cos(theta_C))) # energy of scattered photon
            E_e = E - E_C # deposited by the electron
            E_ab += E_e # add energy of Compton electron
            E = E_C # update gamma energy to continue the while loop

            if e_transport: # electron simulation
                theta_e = Compton_electron(gamma, theta_C)
                theta_e, phi_e = theta_phi_new_frame(theta, phi, theta_e, -phi_C)
                electron(media, geometry, p_forw, theta_e, phi_e, E_e, step) # electron simulation
            else:
                geometry.Edep_update(E_e) # update E_dep matrix
            # update photon direction in reference coordinates system
            theta, phi = theta_phi_new_frame(theta, phi, theta_C, phi_C)
            add_point(p_forw, E_e)
            p_back = p_forw

        else: # coherent scattering
            # calculation of photon scatt. angles
            theta_R = theta_Ray_Sc()
            phi_R = phi_ang()
            E_e = 0. # no electron

            # update photon direction in reference coordinates system
            theta, phi = theta_phi_new_frame(theta, phi, theta_R, phi_R)
            add_point(p_forw, E_e)                
            p_back = p_forw

## electrons
def electron(media, geometry, p_back, theta, phi, E, step):
    z_max = p_back[2]
    e_in = True

    while e_in:
        medium = media[geometry.cur_med]
        e_data = medium.e_data
        index, Edep2, s, mean_scat, tail = e_data.first_step(E)
        if index==-1: # Below first tabulated energy
            geometry.Edep_update(E)
            return e_in, 0., z_max, p_back, theta

        step_list = np.append([[E, Edep2, s, mean_scat, tail]], e_data.step_list[-index:], axis=0)
        for E, Edep2, s, mean_scat, tail in step_list:
            e_in, change, E, p_forw, theta, phi = e_step(
                p_back, E, theta, phi, Edep2, s, mean_scat, tail, geometry, step)
            z = p_forw[2]
            if z>z_max:
                z_max = z
            if not e_in:
                return e_in, E, z_max, p_forw, theta
            p_back = p_forw

            if change:
                # The electron reaches the interface between two media
                # The step list is reinitialized for the electron energy
                # and the new medium (or the same one for backscattered electrons)
                break

        if change:
            continue
        # The electron is absorbed
        return e_in, 0., z_max, p_back, theta

def e_step(p_back, E, theta, phi, Edep2, s, mean_scat, tail, geometry, step):
    change, p_forw, k = step(p_back, theta, phi, s)
    if change:
        # The electron reaches the interface between two media
        # The actual step length is smaller than s
        # Edep, mean_scat and tail are corrected approximately
        Edep = k * 2. * Edep2
        mean_scat *= k
        tail *= k
        s *= k
        # Edep is deposited in the actual voxel and E is updated
        # Then, no energy is deposited on the interface in this step
        # This prevents artifacts on the interface
        geometry.Edep_update(Edep) 
        E -= Edep
        Edep2 = 0.
    else:
        geometry.Edep_update(Edep2) # half Edep energy is deposited in the actual voxel
    
    e_in = geometry.in_out()
    if not e_in:
        # If the electron escapes, only half Edep energy is deposited
        E -= Edep2
        return e_in, change, E, p_forw, theta, phi

    # Check if the angular distribution has a tail (g>0)
    # This tail contribution is assumed to be isotropic
    if tail>0.:
        r = np.random.random()
    else:
        r = 1.
    if r<tail or mean_scat>math.pi:
        theta_scat = theta_isotropic()
    else: # Most steps have gaussian distribution with mean_scat<pi
        theta_scat = abs(np.random.normal(0., mean_scat))

    if theta_scat>=math.pi: # The electron bounces back
        theta = math.pi - theta
        if phi<math.pi:
            phi += math.pi
        else:
            phi -= math.pi
    else:
        phi_scat = phi_ang()
        theta, phi = theta_phi_new_frame(theta, phi, theta_scat, phi_scat) # new theta and phi angles

    if change:
        # The electron was transported to the interface
        # The medium of the next step depends on the propagation direction
        geometry.update_medium(theta, phi)

    # The other half Edep energy is deposited in the final voxel (if change==False)
    geometry.Edep_update(Edep2)
    E -= 2. * Edep2
    return e_in, change, E, p_forw, theta, phi
