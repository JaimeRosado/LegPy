import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources
from . import photon_data as xs
from . import electron_data as ed
#from IPython.display import display
pd.set_option("display.max_rows", None)

N_Av = 6.022e23
alpha = 1./137.036
r_e = 2.8179e-13 #cm

def List_Media(text=None):
    # Show list of media loaded in List_Media.txt as a DataFrame
    # If some text is given, only show entries cointaining this text, ignoring case
    df = pd.read_csv(pkg_resources.open_text(xs,'media_names.txt'), dtype=str, sep='\t')
    df.index += 1
    if text is not None:
        df = df[df.Medium.str.contains(text, case=False)]
    return df
    
class Medium:
    def __init__(self, name=None,  density=None, Pmol=None, N=None, Z=None, A=None, I=None,
                 e_E_min=0.01, e_E_max=1.25):
    # name: name to be given to generic medium or to be searched from nist files
    # density: in g/cm^3. If not given, it is searched from nist files
    # Pmol: molar mass in g
    # N: as [N1,N2,...,Nn], number of atoms of each species on the molecule
    # Z: as [Z1,Z2,...,Zn] atomic number
    # A: as [A1,A2,...,An] mass number
    # I: as [I1,I2,...,In] Bethe parameter in eV
    # example for H2O N=[2,1], Z=[1,8], A=[1,16], I=[19,97]
    # e_E_min: threshold energy in MeV (not used if nist file is found)
    # e_E_max: maximum electron energy to be considered in MeV (not used if nist file is found)
    
        if N is not None:
            N = np.array(N)
        if Z is not None:
            Z = np.array(Z)
        if A is not None:
            A = np.array(A)
        if I is not None:
            I = np.array(I)

        if N is not None and A is not None:
            Pmol_A = (N*A).sum()
            if Pmol is None: # Pmol is calculated from N and A values
                Pmol = Pmol_A

        self.name = name
        ph_data = None
        e_data = None

        # Mandatory parameters to use ph_gen class
        if density is not None and N is not None and Z is not None and Pmol is not None:
            N_elec_mol = (N*Z).sum()
            N_mol_mass = N_Av / Pmol # n of molecules per g

            cs_arr = np.loadtxt(pkg_resources.open_text(xs,'phot_cs_param.txt'))
            b0 = np.interp(Z * alpha, cs_arr[:,0], cs_arr[:,1])
            b1 = np.interp(Z * alpha, cs_arr[:,0], cs_arr[:,2])
            b2 = np.interp(Z * alpha, cs_arr[:,0], cs_arr[:,3])
            f_alpha_Z = np.interp(Z * alpha, cs_arr[:,0], cs_arr[:,4])
            const = 4. * math.pi * r_e**2 * alpha**4 * Z**5 * f_alpha_Z

            # ph_gen class
            ph_data = ph_gen(name, density, N, N_elec_mol, N_mol_mass, const, b0, b1, b2)

        elif name is not None: # ph_nist class
            try:
                cs_nist = np.loadtxt(pkg_resources.open_text(xs, name+'.txt'))
                if density is None: # The user-defined density is used if given
                    density = cs_nist[0,0]
                x_ray_data = pd.read_csv(pkg_resources.open_text(xs, 'x_ray_data.csv'),
                                         dtype=float, delimiter=' ')
                # PZ vector with probability of ionization of each atom Z
                # x_data matrix with data extracted from x_ray_data file for each atom Z
                i = 1
                PZ = []
                PZi = 0.
                x_data = []
                while cs_nist[i,0]>0.: # X ray production data for each atom Z
                    #Z, N, m, kk1, kk2 = cs_nist[i]
                    data = cs_nist[i]
                    PZi += data[1] * data[0]**data[2] # N * Z**m
                    PZ.append(PZi)
                    index = int(data[0])-1 # Z
                    atomi = x_ray_data.loc[index].values[1:] # PY Ekalpha Ekbeta Pkalpha Ebinding
                    x_data.append(atomi)
                    i += 1
                PZ = np.array(PZ)
                PZ /= PZ[-1] # Normalize
                x_data = np.array(x_data)
                cs_nist = cs_nist[i+1:,:] # Cross section data
                # ph_nist class
                ph_data = ph_nist(name, density, cs_nist, PZ, x_data)
            except:
                pass # ph_data = None

        # Mandatory parameters to use e_gen class
        if density is not None and N is not None and Z is not None and A is not None and I is not None:
            w = N * A / Pmol_A # mass fractions. Pmol_A is used even if Pmol is given
            # e_gen class
            e_data = e_gen(name, density, N, Z, A, w, I, e_E_min, e_E_max)

        elif name is not None:  # e_nist class
            try:
                e_data_nist = np.loadtxt(pkg_resources.open_text(ed, name+'.txt'))
                # The user-defined density is used if given
                # If not and it is not loaded from photon nist data
                if density is None:
                    density = e_data_nist[0,0]
                X0 = e_data_nist[0,1] # radiation length in cm
                e_data_nist = e_data_nist[1:]    
                # e_nist class
                e_data = e_nist(name, density, X0, e_data_nist)
            except:
                pass

        if ph_data is None and e_data is None:
            if name is None:
                raise ValueError('Please, input the mandatory parameters.')
            else:
                raise ValueError('Medium not available')
        else:
            self.ph_data = ph_data
            self.e_data = e_data

    def plot_mu(self, energies, l_style='', ph=True, inc=True, coh=True, pair=True, tot=True):
        if self.ph_data is None:
            print('No photon cross section data is available')
            return None
        return Plot_Mu_vs_E(self.ph_data, energies, l_style, ph, inc, coh, pair, tot)

    def plot_R(self, energies, units='cm', l_style = ''):
        if self.e_data is None:
            print('No electron data is available')
            return None
        
        csda = np.interp(energies, self.e_data.E_ref, self.e_data.R_ref)
        y_label = 'R$_{CSDA}$'+'(cm)'  
        if units == 'gcm2':
            csda = csda * self.e_data.density
            y_label = 'R$_{CSDA}$'+'(gcm$^{-2}$)'   
        m_type = self.e_data.name
        s_type = self.e_data.source
        label = m_type + ' ' + s_type        
        plt.loglog(energies, csda, l_style, label = label)
        plt.xlabel('Energy (MeV)')
        plt.ylabel(y_label)
        plt.grid(True, which = 'both')
        plt.title('CSDA Range versus Energy')
        plt.legend()
        #plt.show()

############ Photon data classes
class ph_gen:
    def __init__(self, name, density, N, N_elec_mol, N_mol_mass, const, b0, b1, b2):
        self.source = 'generic'
        self.name = name
        self.density = density
        self.N = N
        self.N_elec_mol = N_elec_mol
        self.N_mol_mass = N_mol_mass
        self.const = const
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.init_E = np.array([[0., None, None, None, None, None]])
        self.E_min = None
        self.E_max = None

    def init_MC(self, energies, pair):
        # Cross section data for uploaded initial energies
        # pair is for pair production. No effect here
        self.init_E = np.array([[0., None, None, None, None, None]]) # Restart initial energies for a new simulation
        if energies is None:
            return None

        for E in energies:
            mu_phot, mu_incoh, mu_coh, mu_pair, mu_total = self.Mu_Cross_section(E)
            self.init_E = np.append(self.init_E, [[E, mu_phot, mu_incoh, mu_coh, mu_pair, mu_total]], axis = 0)

    def Mu_Cross_section(self, E):
        # Check if the input E is an uploaded initial energy and get cross section data if so
        if E in self.init_E[:,0]:
            i = np.searchsorted(self.init_E[:,0], E)
            E, mu_phot, mu_incoh, mu_coh, mu_pair, mu_total = self.init_E[i]
            return mu_phot, mu_incoh, mu_coh, mu_pair, mu_total

        gamma = E / 0.511 #electron mass
        mu_phot = self.Mu_Phot_Cross_section(gamma)
        mu_incoh = self.Mu_Inc_Cross_section(gamma)
        mu_total = mu_phot + mu_incoh
        return mu_phot, mu_incoh, 0., 0., mu_total

    def Rand_track(self, E):
        mean_free_path = 1. / self.Mu_Cross_section(E)[4]
        track = -mean_free_path * math.log(random.random())
        return track / self.density # gcm**-2

    def Rand_proc(self, E):
        mu_phot, mu_incoh, mu_coh, mu_pair, mu_total = self.Mu_Cross_section(E)
        rand_mu = random.random() * mu_total      
        # cross section of pair production goes to the photoelectric one
        if rand_mu <= mu_phot + mu_pair:
            Process = 'Photoelectric'
        elif rand_mu <= mu_phot + mu_incoh + mu_pair:
            Process = 'Compton'
        else:
            Process = 'Coherent'
        return Process

    def Mu_Phot_Cross_section(self, gamma):
        cs_phot_i = self.const * (self.b0 + self.b1 / gamma + self.b2 / gamma / gamma) / gamma #cross section of atom i
        cs_phot = np.dot(self.N, cs_phot_i) # total cross section of molecule
        mu_phot = cs_phot * self.N_mol_mass # mu (g**-1 cm**2)
        return mu_phot

    def Mu_Inc_Cross_section(self, gamma):
        gamma2 = gamma*gamma
        mu_inc = ((1.+gamma)/gamma2*(2.*(1.+gamma)/(1.+2.*gamma)-(math.log(1.+2.*gamma)/gamma)
                  )+math.log(1.+2.*gamma)/(2.*gamma)-(1.+3.*gamma)/(1.+4.*gamma+4.*gamma2)
                  )*4.989344e-25 * self.N_elec_mol * self.N_mol_mass
        return mu_inc


class ph_nist(ph_gen):
    def __init__(self, name, density, cs_nist, PZ, x_data):
        self.source = 'NIST'
        self.name = name
        self.density = density
        self.cs_nist = cs_nist
        self.E_min = cs_nist[0,0]
        self.E_max = cs_nist[-1,0]
        self.PZ = PZ
        self.x_data = x_data

        if np.size(cs_nist, axis=1)==4: # Pair-production cs not loaded on text file
            print("Pair-production cross section not available for this medium. It is set to 0.")
            def zero(E):
                return 0.
            self.Mu_Pair_Cross_section = zero # Overwrite method
        
        self.init_E = np.array([[0., None, None, None, None, None]]) # Restart initial energies for a new simulation

    def init_MC(self, energies, pair=True):
        # Re-define Mu_Pair_Cross_section for a new simulation
        # Cross section data for uploaded initial energies
        if np.size(self.cs_nist, axis=1)==5 and pair:
            def func(E):
                if E<=1.022:
                    return 0.
                mu_pair = np.interp(E, self.cs_nist[:,0], self.cs_nist[:,4])
                return mu_pair
            self.Mu_Pair_Cross_section = func
        else:
            def zero(E):
                return 0.
            self.Mu_Pair_Cross_section = zero
        
        self.init_E = np.array([[0., None, None, None, None, None]]) # Restart initial energies for a new simulation
        if energies is None:
            return None

        for E in energies:
            mu_phot, mu_incoh, mu_coh, mu_pair, mu_total = self.Mu_Cross_section(E)
            self.init_E = np.append(self.init_E, [[E, mu_phot, mu_incoh, mu_coh, mu_pair, mu_total]], axis = 0)

    def Mu_Cross_section(self, E):
        # Check if the input E is an uploaded initial energy and get cross section data if so
        if E in self.init_E[:,0]:
            i = np.searchsorted(self.init_E[:,0], E)
            E, mu_phot, mu_incoh, mu_coh, mu_pair, mu_total = self.init_E[i]
            return mu_phot, mu_incoh, mu_coh, mu_pair, mu_total

        mu_phot = self.Mu_Phot_Cross_section(E)
        mu_incoh = self.Mu_Inc_Cross_section(E)
        mu_coh = self.Mu_Coh_Cross_section(E)
        mu_pair = self.Mu_Pair_Cross_section(E)
        mu_total = mu_phot + mu_incoh + mu_coh + mu_pair
        return mu_phot, mu_incoh, mu_coh, mu_pair, mu_total

    # This process is not implemented yet, but included in the total cross section
    def Mu_Pair_Cross_section(self, E):
        if E<=1.022:
            return 0.
        mu_pair = np.interp(E, self.cs_nist[:,0], self.cs_nist[:,4])
        return mu_pair

    def Mu_Phot_Cross_section(self, E):
        mu_phot = np.interp(E, self.cs_nist[:,0], self.cs_nist[:,3])
        return mu_phot

    def Mu_Inc_Cross_section(self, E):
        mu_inc = np.interp(E, self.cs_nist[:,0], self.cs_nist[:,2])
        return mu_inc

    def Mu_Coh_Cross_section(self, E):
        mu_coh = np.interp(E, self.cs_nist[:,0], self.cs_nist[:,1])
        return mu_coh

    def Xray_prod(self, E):                   
        Rand = np.random.rand()
        index = next((i for i, num in enumerate(self.PZ) if num>Rand), None)
        PZ = self.PZ[index]
        PY, Ekalpha, Ekbeta, Pkalpha, Ebind = self.x_data[index]
        if E>Ebind:
            if np.random.rand()<PY:
                if np.random.rand()<Pkalpha:
                    return Ekalpha
                else:
                    return Ekbeta
        return 0.


def Plot_Mu_vs_E(medium, energies, l_style='', ph=True, inc=True, coh=True, pair=True, tot=True):

    mu_ph = np.empty_like(energies)
    mu_inc = np.empty_like(energies)
    mu_coh = np.empty_like(energies)
    mu_pair = np.empty_like(energies)
    mu_tot = np.empty_like(energies)

    for i, E in enumerate(energies):
        mu_ph[i], mu_inc[i], mu_coh[i], mu_pair[i], mu_tot[i]  = medium.Mu_Cross_section(E)

    # pandas dataframe to excel
    all_mu = np.stack((mu_ph, mu_inc, mu_coh, mu_pair), axis =1)

    s_name = medium.source
    m_type = medium.name

    if ph:
        ph_name = 'Phot' + ' ' + s_name + ' ' + m_type
        line_ph = 'r' + l_style
        plt.loglog(energies, mu_ph, line_ph , label = ph_name)

    if inc:
        inc_name = 'Incoh' + ' ' + s_name + ' ' + m_type
        line_inc = 'b' + l_style
        plt.loglog(energies, mu_inc, line_inc, label = inc_name)

    if coh and s_name == 'NIST':
        coh_name = 'Coh' + ' ' + s_name + ' ' + m_type
        line_coh = 'g' + l_style
        plt.loglog(energies, mu_coh, line_coh, label = coh_name)

    if pair and s_name == 'NIST':
        pair_name = 'Pair' + ' ' + s_name + ' ' + m_type
        line_pair = 'm' + l_style
        plt.loglog(energies, mu_pair, line_pair, label = pair_name)

    if tot:
        tot_name = 'Tot' + ' ' + s_name + ' ' + m_type
        line_tot = 'c' + l_style
        plt.loglog(energies, mu_tot, line_tot, label = tot_name)

    plt.xlabel('Energy (MeV)')
    plt.ylabel('\u03BC'+ '(cm$^2$g$^{-1}$)')
    plt.title('Mass atenuation coeficient (contributions)')
    plt.grid(True, which = 'both')
    plt.legend()

    pd.set_option("display.precision", 3)
    mu_df = pd.DataFrame(all_mu, columns = ['mu_phot', 'mu_inc', 'mu_coh', 'mu_pair'], index = energies*1000.)
    mu_df.index.name = 'Energy (keV)'

    # uncomment to display mu data
    #print('')
    #print('')
    #print('Data from '+s_name+' '+ m_type)
    #display(mu_df)

    # uncomment to write mu data onto a file
    #f_name = 'mu_vs_E_'+s_name+'_'+m_type+'.xlsx'
    #open('mu_vs_E.xlsx', "w") # to excel file
    #mu_df.to_excel(f_name, float_format = '%.3e') # includes bining data
    #print('file '+f_name+'  written to disk')


############ Electron data classes
class e_gen:
    def __init__(self, name, density, N, Z, A, w, I, E_min, E_max):
        self.source = 'generic'
        self.name = name
        self.density = density
        self.E_min = E_min
        self.E_max = E_max
        I = I*1.0e-6  # Bethe parameter in MeV
        X0 = (716.4*A)/(Z*(Z+1.)*np.log(287./(Z**0.5)))/density # radiation length per atom in cm (array)
        self.X0 = 1./(w/X0).sum() # average radiation length in cm

        # Make a list of (Energy, CSDA range) reference data from E_min to E_max, such that E_i = 0.9 * E_i+1
        # Energy list
        log90 = -math.log(0.9)
        logE = np.arange(math.log(E_min), math.log(E_max), log90)
        self.E_ref = np.exp(logE)
        if E_max>self.E_ref[-1]:
            self.E_ref = np.append(self.E_ref, E_max)

        # CSDA list
        self.R_ref = np.zeros_like(self.E_ref)
        R = 0. # CSDA is set to 0 for E=E_min 
        E_lower = E_min
        for j, E_upper in enumerate(self.E_ref[1:]): # Loop over energies
            E = np.linspace(E_lower, E_upper, 100) # Fine energy intervals to integrate the stopping power
            E2 = E*E
            beta2 = 1. / (0.511+E) / (0.511+E)
            T2 = (E2 / 8. - (2.*E+0.511) * 0.511 * 0.69) * beta2
            beta2 = 1. -0.511 * 0.511 * beta2

            E_dep = np.zeros_like(E)
            # Integration of stopping power for each atom
            for Z_a, A_a, I_a, w_a in zip(Z, A, I, w):
                T1 = np.log((E2 * (E+1.02)) / (1.02*I_a*I_a))
                E_dep_a = 0.153 * (Z_a/A_a) * (1./beta2) * (T1+T2+1.-beta2) #MeV cm^2 g^-1
                E_dep += w_a * E_dep_a * density #MeV cm^-1

            # CSDA as a function of energy. Note that R[0]=0
            R += ((E_upper-E_lower)/99./E_dep).sum()
            self.R_ref[j+1] = R
            E_lower = E_upper

    def make_step_list(self, E_cut, length, K, f=None, g=None, h=None, energies=None):
        # Step list with E, E_dep/2, s, mean_scat, tail
        if f is None:
            # Best fit for Z>20 and E=1MeV
            # Backscattering is underestimated in light elements
            f = 0.12983 / self.X0**0.5 + 1.55562
        self.f = f
        if g is None:
            g = 0.
        self.g = g
        if h is None:
            # Energy correction to improve results for light element and/or low energies
            # 0.047 < h, where h is higher for light elements
            h = -0.018182 * np.log( 1. / (self.X0)**0.5) + 0.047091
        self.h = h
        self.E_cut = max(E_cut, self.E_min)
         
        if K is not None: # Constant energy loss fraction
            self.length = None
            self.K = K
            self.k_model()
        else: # Constant step length
            self.length = length*10**-4
            self.K = None
            self.l_model()

        # Mean of gaussian distribution for scattering
        beta2 = 1. / (0.511 + self.E) / (0.511 + self.E)
        betacp = self.E + 0.511 - 0.511 * 0.511 / (0.511 + self.E)
        if self.s[0]==0.:
            mean_scat_gauss = np.zeros_like(self.E)
            mean_scat_gauss[1:] = 13.6 * np.sqrt(self.s[1:]/self.X0) * (1.+0.038*np.log(self.s[1:]/self.X0/beta2)) / betacp[1:]
        else:
            mean_scat_gauss = 13.6 * np.sqrt(self.s/self.X0) * (1.+0.038*np.log(self.s/self.X0/beta2)) / betacp
        k = 1.
        if h!=0.:
            # Default/input f corresponds to reference energy of 1 MeV
            # For E<1MeV, k>1 and this corrections makes the scattering angular distribution broader
            # For E>1MeV, k<1 but the correction keeps small due to the logaritmic dependence
            k = 1. - h * np.log(self.E)
            k[k<0.] = 0. # not needed
            f *= k
        self.k = k
        self.mean_scat = f * mean_scat_gauss

        # Add a "tail" to the angular distribution for a small fraction of isotropic scattering events
        self.tail = g * mean_scat_gauss
        self.tail[self.tail>1.] = 1.
        
        self.N_steps = len(self.E)
        step_list = np.zeros((self.N_steps, 5))
        step_list[:,0] = self.E # E
        step_list[:,1] = np.diff(self.E, prepend=0.) / 2. # E_dep / 2
        step_list[:,2] = np.diff(self.R, prepend=0.) # s
        step_list[:,3] = self.mean_scat # mean_scat
        step_list[:,4] = self.tail # tail
        self.step_list = step_list[::-1,:] # inversed

        self.upload_to_first_steps(energies)

    def k_model(self):
        logK = -math.log(self.K)
        logE = np.arange(math.log(self.E_cut), math.log(self.E_max), logK)
        self.E = np.exp(logE)
        if self.E_max>self.E[-1]:
            self.E = np.append(self.E, self.E_max)

        self.R = np.interp(self.E, self.E_ref, self.R_ref)
        self.s = np.diff(self.R, prepend=0.)

    def l_model(self):
        R_max = self.R_ref[-1]
        R_min = max(self.R_ref[0], self.length)  # =length for e_gen
        self.R = np.arange(R_min, R_max, self.length)
        self.s = np.ones_like(self.R) * self.length
        if R_min>self.length:  # only for e_nist
            self.s[0] = R_min

        top_step = R_max - self.R[-1]
        if top_step>0.:
            self.R = np.append(self.R, R_max)
            self.s = np.append(self.s, top_step)

        self.E = np.interp(self.R, self.R_ref, self.E_ref)
        #E_cut is set to be the energy a which CSDA=R_min
        self.E_cut = self.E[0]

    def upper_energy(self, E):
        #Calculate the index of the closest upper energy in self.E
        #If E>E_max, raise error
        #For E<=E_cut, index = 0
        index = np.searchsorted(self.E, E)
        if index==self.N_steps:
            raise ValueError('Electron energy out of range.')
        return index

    def first_step(self, E):
        # Calculate the length s, deposited energy and mean scattering angle in the first step
        if E<=self.E_cut:
            return -1, None, None, None, None

        if E in self.first_steps[:,0]:
            i = np.searchsorted(self.first_steps[:,0], E)
            E, index, Edep2, s, scat, tail = self.first_steps[i]
            return int(index), Edep2, s, scat, tail

        index = self.upper_energy(E)
        R = np.interp(E, self.E_ref, self.R_ref)
        R_lower = self.R[index-1]
        s = R - R_lower
        E_upper = self.E[index]
        E_lower = self.E[index-1]
        Delta_E = E_upper - E_lower
        E_dep = E - E_lower
        scat_upper = self.mean_scat[index]
        scat_lower = self.mean_scat[index-1]
        Delta_scat = scat_upper - scat_lower
        scat = scat_lower + Delta_scat / Delta_E * E_dep
        tail = min(1., scat * self.g * 13.6 / self.f)
        return index, E_dep/2., s, scat, tail
    
    def upload_to_first_steps(self, energies):
        self.first_steps = np.array([[0., -1, None, None, None, None]]) # restart first_steps list for a new simulation
        if energies is None:
            return None

        for E in energies:
            index, E_dep2, s, scat, tail = self.first_step(E)
            self.first_steps = np.append(self.first_steps, [[E, index, E_dep2, s, scat, tail]], axis = 0)

class e_nist(e_gen):
    def __init__(self, name, density, X0, e_data_nist):
        self.source = 'NIST'
        self.name = name
        self.density = density
        self.X0 = X0
        self.E_ref, self.R_ref = e_data_nist.transpose()
        self.R_ref = self.R_ref / self.density
        self.E_min = self.E_ref[0]
        self.E_max = self.E_ref[-1]