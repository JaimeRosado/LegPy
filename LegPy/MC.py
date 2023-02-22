import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#
from .angles import theta_KN, Compton_electron, theta_isotropic, phi_ang, theta_Ray_Sc, theta_phi_new_frame
from .figures import gamma_hists, e_hists, esc_gammas, fluence
from .analysis_tools import plot_edep_x, plot_edep_y, plot_edep_z
from .geometry import cart_vox

def Plot_beam(media, geometry, spectrum, beam, n_part=50, E_cut=0.01,
              tracks=True, points=False,
              e_length=None, e_K=None, e_f=None, e_g=0.):
    MC(media, geometry, spectrum, beam, n_part=n_part, E_cut=E_cut,
       e_transport=False, tracks=tracks, points=points,
       e_length=e_length, e_K=e_K, e_f=e_f, e_g=e_g)

##### MC options
class MC:
    def __init__(self, media, geometry, spectrum, beam,
                 n_part=1000, E_cut=0.01, n_ang=20, n_E=20, n_z=20,
                 e_transport=False, tracks=False, points=False, gamma_out=False,
                 e_length=None, e_K=None, e_f=None, e_g=0.):

        # media may be a list or a Medium object
        if isinstance(media, list):
            N_media = len(media)
        else:
            media = [media]
            N_media = 1
        # Error handling for media
        if geometry.N_media<N_media:
            N_media = geometry.N_media
            #print('The input geometry only accepts {} media.'.format(N_media),
            #      'Only the first {} media are used.'.format(N_media))
            print('The input geometry only accepts one medium. Only the first input medium is used.')
            media = media[0:N_media]
        elif geometry.N_media>N_media:
            raise ValueError('The input geometry expects 2 media, but only 1 medium is given.')
        self.media = media
        self.N_media = N_media

        # Set default file name
        fname = ''
        for index, medium in enumerate(media):
            fname = fname + medium.name
            if index<N_media-1:
                fname = fname + '_'
        self.fname = fname
        
        self.geometry = geometry
        self.spectrum = spectrum
        E_max = spectrum.E_max
        self.E_max = E_max
        self.beam = beam
        part_type = beam.particle
        self.part_type = part_type
        self.n_part = n_part
        self.E_cut = E_cut
        self.e_transport = e_transport
        self.e_length = e_length # electron step length in um
        self.e_K = e_K # 1-e_k is the decimal fraction of energy loss per step
        self.e_f = e_f # model parameter for gaussian angular distribution for elect. scattering
        self.e_g = e_g # model parameter for the isotropic component of the angular distribution

        # Initial particle energies for which calculations are speeded up
        if spectrum.name=='mono':
            energies = np.array([spectrum.E])
        elif spectrum.name=='multi_mono':
            energies = np.sort(spectrum.energies)
        else:
            energies = None

        # Electron beam or photon beam with electron transport included
        if part_type=='electron' or e_transport:
            # Error handling for e_data
            for index, medium in enumerate(media):
                e_data = medium.e_data
                if e_data is None:
                    raise ValueError('No electron data is available for the medium {}'.format(index+1))
                if e_data.E_max<spectrum.E_max:
                    raise ValueError(
                        'Maximum electron energy out of range of the electron data loaded to the medium {}'.format(index+1))

                # By default, e_length is used (instead of e_K) and set to the minimum voxel size
                vox_length = geometry.voxelization.min_delta()
                if e_length is None and e_K is None:
                    e_length = vox_length
                    self.e_length = e_length
                
                # The electron step list is generated for each medium according to
                # the simulation options and considering the usual energies
                e_data.make_step_list(E_cut, e_length, e_K, e_f, e_g, energies)

                # If the default e_length value is not used
                # warn if the step length is larger than the voxel size
                # The cm-to-um conversion factor is taken as 9999 instead of 10000
                # to avoid some round issues in e_data.s
                if e_data.s.max()*9999.>vox_length:
                    print('The step length of some electrons may be greater than the voxel size.')

        # Photon beam
        if part_type!='electron':
            if E_max>1.022: # Pair production is included
                self.pair = True
            else:
                self.pair = False

            # Error handling for ph_data
            for index, medium in enumerate(media):
                ph_data = medium.ph_data
                if ph_data is None:
                    raise ValueError('No photon data is available for the medium {}'.format(index+1))
                if ph_data.E_max is not None: # NIST medium
                    if ph_data.E_max<spectrum.E_max:
                        raise ValueError(
                            'Maximum photon energy out of range of the cross section data loaded to the medium {}'.format(index+1))

                # Prepare Mu_Pair_Cross_section and load data for initial energies
                ph_data.init_MC(energies, self.pair)

        # Init Edep matrix
        geometry.Edep_init()

        # Redefine part_step for 2 media
        if N_media>1:
            self.part_step = self.part_step_2M

        # Add track visualization to part_step
        if tracks:
            self.ax = geometry.plot()
            self.part_step = self.add_track(self.part_step)

        # The function particle and histograms are defined according to the primary particle
        if part_type=='electron': # electron beam
            particle = self.electron
            # Histograms of both maximum and final z as well as of angle of backscattered electrons
            self.hists = e_hists(n_z, n_ang, geometry.z_top, n_part)
            # Angle vs E plot can only be produced for photon beams
            gamma_out = False
            self.add_gamma_out = self.nothing

        else: # photon beam
            particle = self.photon
            # Fluence histogram
            self.fluence = fluence(geometry, n_z, n_E, E_max, n_part)
            # Histograms of absorbed energy as well as of both energy and angle of escaped photons
            self.hists = gamma_hists(n_ang, n_E, E_max, n_part)
            if gamma_out:
                # Plot of theta vs energy of escaped photons
                self.gamma_out = esc_gammas(E_max)
            else:
                self.add_gamma_out = self.nothing
            # Energy deposit visualization in figure of tracks
            if points:
                if not tracks:
                    self.ax = geometry.plot()
            else:
                self.add_point = self.nothing

        # Computing time (Start)
        time_i = timeit.default_timer()

        # loop over particles
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
            output = particle(position, theta, phi, E)
            self.hists.add_count(output)
            self.add_gamma_out(output)

        if not tracks and not points and not gamma_out:
            time_f = timeit.default_timer() # Computing time (End)
            time_per_part = (time_f - time_i) / (n_part)
            print()
            print('The simulation has ended')
            print()
            print('Computing time per beam particle = ','{:.2e}'.format(time_per_part), 'seconds')
            print()

        self.Edep = geometry.Edep_out(n_part)
        # Angle vs E plot (only produced for photon beams)
        if gamma_out:
            if n_part>1000:
                print('Number of photons is recommended to be less than 1000.')
            self.gamma_out.plot()

    def plot_Edep(self):
        self.geometry.Edep_plot(self.Edep)

    def plot_Edep_layers(self, indexes, axis='z', c_profiles=False, lev=None):
        if not isinstance(self.geometry.voxelization, cart_vox):
            raise ValueError('This tool is only available for cartesian voxelization.')
        Edep = self.Edep
        x = self.geometry.x
        y = self.geometry.y
        z = self.geometry.z
        n_x = self.geometry.voxelization.n_x
        n_y = self.geometry.voxelization.n_y
        n_z = self.geometry.voxelization.n_z
        if axis=='x':
            plot_edep_x(Edep, x, y, z, n_z, n_x, n_y, indexes, c_profiles=c_profiles, lev=lev)
        elif axis=='y':
            plot_edep_y(Edep, x, y, z, n_z, n_x, n_y, indexes, c_profiles=c_profiles, lev=lev)
        else:
            plot_edep_z(Edep, x, y, z, n_z, n_x, n_y, indexes, c_profiles=c_profiles, lev=lev)

    def Edep_to_npy(self, fname=None):
        if fname is None:
            fname = self.fname
        fname = fname + '.npy' 
        open(fname, 'w')
        np.save(fname, self.Edep)

    def Edep_to_df(self):
        return self.geometry.Edep_to_df(self.Edep)

    def Edep_to_excel(self, fname=None):
        if fname is None:
            fname = self.fname
        Edep_df = self.Edep_to_df()
        self.geometry.Edep_to_excel(Edep_df, fname)

    def hists_to_excel(self, fname=None):
        if self.part_type=='electron':
            raise ValueError('This option is not implemented for electron beams yet.')
        if fname is None:
            fname = self.fname
        self.hists.to_excel(fname)

    def plot_hists(self):
        self.hists.plot()

    def final_z(self):
        if not self.part_type=='electron':
            raise ValueError('Histogram only available for electron beams.')
        return self.hists.final_z()

    def max_z(self):
        if not self.part_type=='electron':
            raise ValueError('Histogram only available for electron beams.')
        return self.hists.max_z()

    def ext_range(self, definition='final'):
        if not self.part_type=='electron':
            raise ValueError('Histogram only available for electron beams.')
        return self.hists.ext_range(definition)

    def backscattering(self):
        if not self.part_type=='electron':
            raise ValueError('Histogram only available for electron beams.')
        return self.hists.backscattering()

    def ang_out(self):
        if self.part_type=='electron':
            raise ValueError('Histogram only available for photon beams.')
        return self.hists.ang_out()

    def E_out(self):
        if self.part_type=='electron':
            raise ValueError('Histogram only available for photon beams.')
        return self.hists.E_out()

    def E_ab(self):
        if self.part_type=='electron':
            raise ValueError('Histogram only available for photon beams.')
        return self.hists.E_ab()

    def plot_fluence(self):
        if self.part_type=='electron':
            raise ValueError('Option only available for photon beams.')
        self.fluence.plot()

    def fluence_to_df(self):
        if self.part_type=='electron':
            raise ValueError('Option only available for photon beams.')
        return self.fluence.to_df()

    def fluence_to_excel(self, fname=None):
        if self.part_type=='electron':
            raise ValueError('Option only available for photon beams.')
        if fname is None:
            fname = self.fname
        self.fluence.to_excel(fname)

    def take_step(self, p_back, theta, phi, step_length):
        # Take a step of length step_length from p_back in the direction theta, phi
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        # update particle direction
        Direction = np.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]) #unit vector
        return p_back + step_length * Direction # step

    def part_step(self, p_back, theta, phi, step_length):
        # Particle step in a single medium
        # change is set to False
        # step_length is unchanged
        p_forw = self.take_step(p_back, theta, phi, step_length)
        self.geometry.try_position(p_forw)
        return False, p_forw, 1.

    def part_step_2M(self, p_back, theta, phi, step_length):
        # Particle step for two media
        # Check if the particle changes of medium
        # If so, the particle is transported to the interface and p_forw and step_length are updated
        p_forw = self.take_step(p_back, theta, phi, step_length)
        change, p_forw, k = self.geometry.update_position(p_forw, step_length)
        return change, p_forw, k

    def add_track(self, part_step):
        # Include track visualization to part_step
        def func(p_back, theta, phi, step_length):
            change, p_forw, k = part_step(p_back, theta, phi, step_length)
            self.geometry.tracks(p_back, p_forw, self.ax)
            return change, p_forw, k
        return func
    
    def add_point(self, position, Edep):
        self.geometry.points(position, Edep, self.E_max, self.ax)

    def add_gamma_out(self, output):
        self.gamma_out.add_count(output)

    def plot_gamma_out(self, n_part=1000):
        if self.part_type=='electron':
            raise ValueError('Tool not available for electron beams.')
        result = MC(self.media, self.geometry, self.spectrum, self.beam,
                 n_part=n_part, E_cut=self.E_cut, gamma_out=True)

    def nothing(self, *arg):
        pass

    def photon(self, p_back, theta, phi, E):
        geometry = self.geometry
        medium = self.media[geometry.cur_med]
        ph_data = medium.ph_data
        E_ab = 0. # initialize absorbed energy for the new photon
        phot_in = True
        new_phot = True
        while phot_in:
            if E <= self.E_cut:
                geometry.Edep_update(E) # update E_dep matrix
                E_ab += E # add energy of photon
                return False, E_ab, E, theta

            # track length for photon energy
            step_length = ph_data.Rand_track(E)
            change, p_forw, k = self.part_step(p_back, theta, phi, step_length)
            self.fluence.add_count(p_back, p_forw, k*step_length, E)

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
                geometry.cur_med = 1 - geometry.cur_med
                medium = self.media[geometry.cur_med]
                ph_data = medium.ph_data # Change medium
                p_back = p_forw
                continue

            new_phot = False # The photon interacts
            Proc = ph_data.Rand_proc(E)
            if Proc == 'Photoelectric':
                E_ab += E # add photoelectric absorption.
                if self.e_transport:
                    self.electron(p_forw, theta, phi, E)
                    # After simulating the electron, resume the photon
                    geometry.try_position(p_forw)
                    part_in = geometry.in_out()
                    geometry.init_medium(theta, phi)
                else:
                    geometry.Edep_update(E)
                self.add_point(p_forw, E)
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

                if self.e_transport: # electron simulation
                    theta_e = Compton_electron(gamma, theta_C)
                    theta_e, phi_e = theta_phi_new_frame(theta, phi, theta_e, -phi_C)
                    self.electron(p_forw, theta_e, phi_e, E_e) # electron simulation
                    # After simulating the electron, resume the photon
                    geometry.try_position(p_forw)
                    part_in = geometry.in_out()
                    geometry.init_medium(theta, phi)
                else:
                    geometry.Edep_update(E_e) # update E_dep matrix
                # update photon direction in reference coordinates system
                theta, phi = theta_phi_new_frame(theta, phi, theta_C, phi_C)
                self.add_point(p_forw, E_e)
                p_back = p_forw

            else: # coherent scattering
                # calculation of photon scatt. angles
                theta_R = theta_Ray_Sc()
                phi_R = phi_ang()
                E_e = 0. # no electron

                # update photon direction in reference coordinates system
                theta, phi = theta_phi_new_frame(theta, phi, theta_R, phi_R)
                self.add_point(p_forw, E_e)                
                p_back = p_forw

    def electron(self, p_back, theta, phi, E):
        geometry = self.geometry
        z_max = p_back[2]
        e_in = True

        while e_in:
            medium = self.media[geometry.cur_med]
            e_data = medium.e_data
            index, Edep2, s, mean_scat, tail = e_data.first_step(E)
            if index==-1: # Below first tabulated energy
                geometry.Edep_update(E)
                return e_in, 0., z_max, p_back, theta

            step_list = np.append([[E, Edep2, s, mean_scat, tail]], e_data.step_list[-index:], axis=0)
            for E, Edep2, s, mean_scat, tail in step_list:
                e_in, change, E, p_forw, theta, phi = self.e_step(
                    p_back, E, theta, phi, Edep2, s, mean_scat, tail)
                z = p_forw[2]
                if z>z_max:
                    z_max = z
                if not e_in:
                    return e_in, E, z_max, p_forw, theta
                p_back = p_forw

                if change:
                    # The electron reaches the interface between two media
                    # The step list is reinitialized for the new medium
                    # with the electron having energy E and being at p_forw with direction theta, phi
                    break

            if change:
                continue
            # The electron is absorbed
            return e_in, 0., z_max, p_back, theta

    def e_step(self, p_back, E, theta, phi, Edep2, s, mean_scat, tail):
        geometry = self.geometry
        change, p_forw, k = self.part_step(p_back, theta, phi, s)
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
