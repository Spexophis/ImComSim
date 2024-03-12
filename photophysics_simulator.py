import numpy as np
from numpy import linalg


class Experiment:
    def __init__(self, illumination, fluorophore):

        self.illumination = illumination
        self.fluorophore = fluorophore

    def solve_kinetics(self, time_step):
        """Solve the time evolution of the proposed kinetic model for a pulse scheme"""

        # Calculate the illumination scheme from the illumination class. The illumination scheme contains the pulse
        # modulation of the operating laser in the experiments with its corresponding photon flux calculated from the
        # operating wavelengths. Calculate also the width of the time windows.
        photon_flux, pulse_windows, laser_modulation = self.illumination.excitationScheme()

        # Get the kinetic model matrix for the proposed pulse scheme
        laser_scheme = np.transpose(np.multiply(np.transpose(laser_modulation), self.fluorophore.wavelength))
        km = self.fluorophore.kineticModel(photon_flux, laser_scheme)

        # Get the initial populations from the fluorophore class.
        p0 = self.fluorophore.initialCondition()

        # Get total pulse dwell time to determine the number of simulated time points
        time_lab = np.cumsum(pulse_windows)
        time_lab = np.insert(time_lab, [0], 0)
        pulse_length = self.illumination.dwellTime
        time_points = np.arange(0, pulse_length, time_step)

        # Initialize population list
        p = np.zeros((time_points.size, self.fluorophore.nstates))
        store_p0 = np.zeros((pulse_windows.size, self.fluorophore.nstates))

        # Solve time evolution in every time window
        for i in np.arange(pulse_windows.size):
            time_sel = np.array((int(round(time_lab[i] / time_step, 2)), int((round(time_lab[i + 1] / time_step, 2)))))
            # time_w = np.arange(timeStep, round((pulseWindows[i]+timeStep),1), timeStep)
            time_w = np.arange(time_step, round((time_sel[-1] + 1 - time_sel[0]) * time_step, 2), time_step)
            if np.size(time_w) != round(time_sel[-1] - time_sel[0]):
                time_w = time_w[:-1]
            else:
                time_w = time_w
            populations_window, _, _ = self.time_evolution(km[i, :, :], p0, time_w)
            p[time_sel[0]:time_sel[1], :] = populations_window
            p0 = populations_window[-1, :]
            store_p0[i, :] = p0

        return p

    @staticmethod
    def time_evolution(M, p0, time):
        """Population's time-evolution solver"""
        # Initialize variables
        rang = int(np.shape(M)[0])

        # Solve the kinetic problem analytically for a matrix M
        Lambda, Q = linalg.eig(M)
        Qinv = linalg.inv(Q)
        timeExponential = np.exp(time[:, np.newaxis] * Lambda)
        timePropagation = timeExponential[:, :, np.newaxis] * np.eye(rang)
        timeEvolve = np.matmul(np.matmul(Q, timePropagation), Qinv)
        populations_t = np.matmul(timeEvolve, p0)

        return populations_t, Lambda, Q


class ModulatedLasers:
    def __init__(self, wavelengths, power_densities, pulse_widths, t_start, dwell_time):

        self.lambdas = np.array(wavelengths)  # nm
        self.powerDensities = np.array(power_densities)  # kW/cm2
        self.pulseWidthLambda = np.array(pulse_widths)  # ms
        self.tStartLambda = np.array(t_start)  # ms
        self.dwellTime = np.array(dwell_time)  # ms
        self.illuminationScheme, self.timeWindows, self.laserModulation = self.excitationScheme()
        self.photonFlux = self.calculatePhotonFlux(self.lambdas, self.powerDensities)

    def calculatePhotonFlux(self, wavelength, powerDensity):
        # Compute the photon flux in photons/(cm2 ms)
        c = 3E8  # speed of light (m/s)
        h = 6.63E-34  # Planck's constant (Js)
        photonEnergy = c * h / (wavelength * 1E-9)
        photonFlux = np.divide(powerDensity, photonEnergy)
        photonFlux = np.diag(photonFlux)
        return photonFlux

    def pulseScheme(self, pulseWidths, tStart, dwellTime):
        # Initialize variables
        pulse = np.zeros((np.size(tStart, 0), 2))

        # Define the pulse according to the input parameters. Create a pulse scheme array vector with the starting and
        # ending times for every wavelength.
        tEnd = np.add(tStart, pulseWidths)

        for x in range(0, np.size(tStart)):
            pulse[x] = [tStart[x], tEnd[x]]
        pulse = np.reshape(pulse, (1, np.size(tStart, 0) * 2))

        # Every wavelength is given a number label. Define the events of turning ON & OFF every laser. Sorting the pulse
        # variable recreates the sequences of ON & OFF lasers in the lab and these are sorted accordingly as every
        # laser has its corresponding label.
        pulseON = np.zeros(np.size(pulse))
        pulseOFF = np.zeros(np.size(pulse))

        laserN = np.arange(1, (np.size(pulse, 1) / 2) + 1, 1)

        selON = np.arange(0, np.size(pulse), 2)
        selOFF = np.arange(1, np.size(pulse), 2)

        pulseON[selON] = laserN
        pulseOFF[selOFF] = laserN

        pulseSort = np.sort(pulse)
        indxs = np.argsort(pulse)

        pulseONsort = pulseON[indxs]
        pulseOFFsort = pulseOFF[indxs]

        # Use some logic operations to create the laser modulation matrix. In this matrix, every row corresponds to a laser
        # and every column represents a so-called kinetic window. Every kinetic window is defined by the number of lasers
        # that are on (or off) for that instance and for how long.
        laserMod = np.zeros((np.size(laserN), np.size(pulseSort) + 1))

        for x in range(0, np.size(pulseSort)):
            laserMod[:, x + 1] = laserMod[:, x]
            pONsBool = int(pulseONsort[0, x])
            pOFFsBool = int(pulseOFFsort[0, x])
            if pONsBool != 0:
                laserMod[pONsBool - 1, x + 1] = 1
            if pOFFsBool != 0:
                laserMod[pOFFsBool - 1, x + 1] = 0

        # Time window matrix represents the length in time of every kinetic window. Important to propagate the population
        # for every kinetic window. Start from a lab time axis.Time window matrix contains in each element the
        # length of the events inside that window.
        tLab = np.zeros(np.size(pulseSort) + 2)
        tLab[1:-1] = pulseSort
        tLab[-1] = dwellTime

        tWindowM = np.zeros(np.size(laserMod, 1))

        for x in range(0, np.size(tWindowM)):
            tWindowM[x] = tLab[x + 1] - tLab[x]

        # Certain pulse schemes can give meaningless time modulations with time windows of length equal to 0. Examples
        # are if two signals end at the same time or the tEnd of one of the signals coincides with the end of pulse. The
        # variable cond1 accounts for these timeWindows equal to 0 and allows to delete them from the arrays.

        cond1 = np.where(tWindowM == 0)

        tWindowM = np.delete(tWindowM, cond1, 0)
        laserMod = np.delete(laserMod, cond1, 1)

        # Return the laser modulation matrix and the time window matrix.
        return laserMod, tWindowM

    def excitationScheme(self):
        # Compute the photon flux from the given power densities and wavelengths
        photonFlux = ModulatedLasers.calculatePhotonFlux(self, self.lambdas, self.powerDensities)

        # Compute the illumination scheme for an experiment. Define the illumination scheme as the photon flux for every
        # wavelength in every kinetic window
        laserModulation, timeWindows = ModulatedLasers.pulseScheme(self, self.pulseWidthLambda, self.tStartLambda,
                                                                   self.dwellTime)
        illuminationScheme = np.matmul(photonFlux, laserModulation)

        return illuminationScheme, timeWindows, laserModulation


class NegativeSwitchers:
    def __init__(self,
                 extincion_coeff_on=[5260, 51560],
                 extincion_coeff_off=[22000, 60],
                 wavelength=[405, 488],
                 lifetime_on=1.6E-6,
                 lifetime_off=20E-9,
                 qy_cis_to_trans_anionic=1.65e-2,
                 qy_trans_to_cis_anionic=2e-2,
                 qy_fluorescence_on=0.35,
                 initial_populations=[0, 0, 1, 0],
                 extincion_coeff_triplet=[0, 0, 0],  # rsFP + triplet and bleaching
                 lifetime_triplet=5.,  # rsFP + triplet and bleaching
                 lifetime_triplet_excited=1e-9,  # rsFP + triplet and bleaching
                 inter_system_crossing=2.7e-3,  # rsFP + triplet and bleaching [Rane, 2023]
                 reverse_inter_system_crossing=1.2E-3,  # rsFP + triplet and bleaching [Rane, 2023]
                 qy_bleaching_on=0.,  # rsFP + triplet and bleaching,
                 qy_bleaching_off=0.,  # rsFP + triplet and bleaching
                 qy_bleaching_triplet=0.,  # rsFP + triplet +bleaching
                 qy_bleaching_triplet_exc=0.,
                 lifetime_prot_trans=48e-3,  # rsFP + triplet and bleaching [Woodhouse, 2020]
                 lifetime_deprot_cis=825e-3,  # rsFP + triplet and bleaching [Woodhouse, 2020]
                 pKa_cis=5.9,  # rsFP + triplet and bleaching [El Khatib, 2015]
                 pH=7.5,  # rsFP + triplet and bleaching
                 qy_cis_to_trans_neutral=0,  # rsFP + triplet [Volpato, 2023]
                 qy_trans_to_cis_neutral=0.33,  # rsFP + triplet [El Khatib, 2015]
                 lifetime_maturation=5.1e-3,  # rsFP complete model [Woodhouse, 2020]
                 extincion_coeff_immature=[7000, 220000, 0],  # rsFP complete model [Woodhouse, 2020]
                 qy_im_to_off=1.65e-2,  # rsFP complete model [Woodhouse, 2020]
                 qy_fluorescence_im=0.35,  # rsFP complete model [Woodhouse, 2020]
                 nspecies=4):

        self.wavelength = np.array(wavelength)
        self.extincion_coeff_on = np.array(extincion_coeff_on)
        self.extincion_coeff_off = np.array(extincion_coeff_off)
        self.extincion_coeff_triplet = np.array(extincion_coeff_triplet)
        self.extincion_coeff_immature = np.array(extincion_coeff_immature)

        # pH & pKas
        self.pH = pH
        self.pKa_cis = pKa_cis

        # Cross-section of absroption in cm2
        epsilon2sigma = 3.825e-21  # [Tkachenko2007, page 5]
        self.cross_section_on = self.extincion_coeff_on * epsilon2sigma
        self.cross_section_off = self.extincion_coeff_off * epsilon2sigma
        self.cross_section_triplet = self.extincion_coeff_triplet * epsilon2sigma
        self.cross_section_immature = self.extincion_coeff_immature * epsilon2sigma

        # Lifetime of states in miliseconds
        self.tau_on = lifetime_on
        self.tau_off = lifetime_off
        self.tau_triplet = lifetime_triplet
        self.tau_triplet_excited = lifetime_triplet_excited
        self.tau_prot_trans = lifetime_prot_trans
        self.tau_deprot_cis = lifetime_deprot_cis
        self.tau_prot_cis = self.tau_deprot_cis * np.power(10, (self.pH - self.pKa_cis))
        self.tau_maturation = lifetime_maturation

        # Quantum yields
        self.qy_cis_to_trans_anionic = qy_cis_to_trans_anionic
        self.qy_trans_to_cis_anionic = qy_trans_to_cis_anionic
        self.qy_fluorescence_on = qy_fluorescence_on
        self.qy_isc = inter_system_crossing
        self.qy_re_isc = reverse_inter_system_crossing
        self.qy_bleaching_on = qy_bleaching_on
        self.qy_bleaching_off = qy_bleaching_off
        self.qy_bleaching_triplet = qy_bleaching_triplet
        self.qy_bleaching_triplet_exc = qy_bleaching_triplet_exc
        self.qy_cis_to_trans_neutral = qy_cis_to_trans_neutral
        self.qy_trans_to_cis_neutral = qy_trans_to_cis_neutral
        self.qy_off_im = qy_im_to_off
        self.qy_fluorescence_im = qy_fluorescence_im

        # Number of states in the kinetic model
        self.nstates = nspecies

        # Index of the fluorescent state. If simulating the complete model include
        # the fluorescence qy of that state, otherwise just the cis-anionic fluorescence qy
        self.qy_fluo = np.zeros((self.nstates))
        if self.nstates == 13:
            self.qy_fluo[3] = qy_fluorescence_on
            self.qy_fluo[-1] = qy_fluorescence_im
        else:
            self.qy_fluo[3] = qy_fluorescence_on

        # Initial conditions
        self.initial_populations = initial_populations

    def kineticModel(self, PF, lasers):
        # Compute the kinetic matrix. Order of the lasers in F
        # should match the order wavelengths in vector lasers. In
        # general take F[0] as On switching light and F[1] as Excitation light

        # Initialize arrays
        PFeye = np.ones(PF.shape[1])
        K = np.zeros((PF.shape[1], self.nstates, self.nstates))

        # Connect the k-rates with the right species
        nlasers = PF.shape[1]
        nwavelengths = self.wavelength.size

        # 4 states rsFP
        # States labels: 0-trans0, 1-trans1, 2-cis0, 3-cis1

        # Absorption processes
        if self.nstates == 4:
            self.fluorophore_type = 'rsFP_4states'
            for i in range(nlasers):
                for j in range(nwavelengths):
                    if lasers[j, i] == self.wavelength[j]:
                        K[:, 1, 0] = K[:, 1, 0] + PF[j] * self.cross_section_off[j]  # trans0_neutral absorption
                        K[:, 3, 2] = K[:, 3, 2] + PF[j] * self.cross_section_on[j]  # cis0_anionic absorption

            # Radiative & non-radiative decays
            K[:, 0, 1] = PFeye / self.tau_off  # Off state lifetime
            K[:, 2, 3] = PFeye / self.tau_on  # On state lifetime

            # Switching processes
            K[:, 2, 1] = PFeye * (self.qy_trans_to_cis_neutral / self.tau_off)  # On switching
            K[:, 0, 3] = PFeye * (self.qy_cis_to_trans_anionic / self.tau_on)  # On switching

        ############################################################################################################
        # 11 states rsFP. Eight-states model system + triplet + bleaching

        # States labels: 0-trans0_anionic, 1-trans1_anionic, 2-cis0_anionic, 3-cis1_anionic, 4-triplet0, 5-triplet1,
        # 6-bleached, 7-trans0_neutral, 8-trans1_neutral, 9-cis0_neutral, 10-cis1_neutral

        # Absorption processes
        if self.nstates == 11:
            self.fluorophore_type = 'rsFP_8states_triplet_bleaching'
            for i in range(nlasers):
                for j in range(nwavelengths):
                    if lasers[j, i] == self.wavelength[j]:
                        K[:, 1, 0] = K[:, 1, 0] + PF[j] * self.cross_section_on[j]  # trans0_anionic absorption
                        K[:, 3, 2] = K[:, 3, 2] + PF[j] * self.cross_section_on[j]  # cis0_anionic absorption
                        K[:, 8, 7] = K[:, 8, 7] + PF[j] * self.cross_section_off[j]  # trans0_neutral absorption
                        K[:, 10, 9] = K[:, 10, 9] + PF[j] * self.cross_section_off[j]  # cis0_neutral absorption
                        K[:, 5, 4] = K[:, 5, 4] + PF[j] * self.cross_section_triplet[j]  # triplet absorption

            # Radiative & non-radiative decays (fluorescence or excited state relaxations)
            K[:, 0, 1] = PFeye / self.tau_on  # On state lifetime
            K[:, 2, 3] = PFeye / self.tau_on  # On state lifetime
            K[:, 7, 8] = PFeye / self.tau_off  # Off state lifetime
            K[:, 9, 10] = PFeye / self.tau_off  # Off state lifetime
            K[:, 4, 5] = PFeye / self.tau_triplet_excited  # Triplet excited state lifetime

            # Switching processes (transitions between states, also triplet and bleaching)
            K[:, 2, 1] = PFeye * (self.qy_trans_to_cis_anionic / self.tau_on)  # Trans to cis anionic
            K[:, 0, 3] = PFeye * (self.qy_cis_to_trans_anionic / self.tau_on)  # Off switching
            K[:, 4, 3] = PFeye * (self.qy_isc / self.tau_on)  # Triplet state formation
            K[:, 3, 5] = PFeye * (
                    self.qy_re_isc / self.tau_triplet_excited)  # Light induced relaxation of triplet state
            K[:, 2, 4] = PFeye / self.tau_triplet  # Triplet state "thermal" relaxation
            K[:, 6, 4] = PFeye * (self.qy_bleaching_triplet / self.tau_triplet)  # Bleaching from triplet0
            K[:, 6, 5] = PFeye * (self.qy_bleaching_triplet / self.tau_triplet_excited)  # Bleaching from triplet1
            K[:, 9, 8] = PFeye * (self.qy_trans_to_cis_neutral / self.tau_off)  # On switching
            K[:, 7, 10] = PFeye * (self.qy_cis_to_trans_neutral / self.tau_off)  # Cis to trans neutral
            K[:, 6, 8] = PFeye * (self.qy_bleaching_off / self.tau_off)  # Bleaching from trans neutral
            K[:, 6, 10] = PFeye * (self.qy_bleaching_off / self.tau_off)  # Bleaching from cis neutral

            # Protonation & deprotonation processes
            K[:, 7, 0] = PFeye / self.tau_prot_trans  # Trans state protonation
            K[:, 9, 2] = PFeye / self.tau_prot_cis  # Cis state protonation
            K[:, 2, 9] = PFeye / self.tau_deprot_cis  # Cis state deprotonation

        ############################################################################################################
        # 7 states rsFP. Four-states model system + triplet + bleaching

        # States labels: 0-trans0_anionic, 1-trans1_anionic, 2-cis0_anionic, 3-cis1_anionic, 4-triplet0, 5-triplet1,
        # 6-bleached

        # Absorption processes
        if self.nstates == 7:
            self.fluorophore_type = 'rsFP_4states_triplet_bleaching'
            for i in range(nlasers):
                for j in range(nwavelengths):
                    if lasers[j, i] == self.wavelength[j]:
                        K[:, 1, 0] = K[:, 1, 0] + PF[j] * self.cross_section_off[j]  # trans0_anionic absorption
                        K[:, 3, 2] = K[:, 3, 2] + PF[j] * self.cross_section_on[j]  # cis0_anionic absorption
                        K[:, 5, 4] = K[:, 5, 4] + PF[j] * self.cross_section_triplet[j]  # triplet absorption

            # Radiative & non-radiative decays (fluorescence or excited state relaxations)
            K[:, 0, 1] = PFeye / self.tau_off
            K[:, 2, 3] = PFeye / self.tau_on
            K[:, 4, 5] = PFeye / self.tau_triplet_excited

            # Switching processes (transitions between states, also triplet and bleaching)
            K[:, 2, 1] = PFeye * (self.qy_trans_to_cis_neutral / self.tau_off)
            K[:, 0, 3] = PFeye * (self.qy_cis_to_trans_anionic / self.tau_on)
            K[:, 4, 3] = PFeye * (self.qy_isc / self.tau_on)
            K[:, 3, 5] = PFeye * (self.qy_re_isc / self.tau_triplet_excited)
            K[:, 2, 4] = PFeye / self.tau_triplet
            K[:, 6, 4] = PFeye * (self.qy_bleaching_triplet / self.tau_triplet)

        ############################################################################################################
        # States labels: 0-trans0_anionic, 1-trans1_anionic, 2-cis0_anionic, 3-cis1_anionic, 4-triplet0, 5-triplet1,
        # 6-bleached, 7-trans0_neutral, 8-trans1_neutral, 9-cis0_neutral, 10-cis1_neutral, 11-cis0_immature, 12-cis1_immature

        # Absorption processes
        if self.nstates == 13:
            self.fluorophore_type = 'rsFP_10states_triplet_bleaching'
            for i in range(nlasers):
                for j in range(nwavelengths):
                    if lasers[j, i] == self.wavelength[j]:
                        K[:, 1, 0] = K[:, 1, 0] + PF[j] * self.cross_section_on[j]  # trans0_anionic absorption
                        K[:, 3, 2] = K[:, 3, 2] + PF[j] * self.cross_section_on[j]  # cis0_anionic absorption
                        K[:, 8, 7] = K[:, 8, 7] + PF[j] * self.cross_section_off[j]  # trans0_neutral absorption
                        K[:, 10, 9] = K[:, 10, 9] + PF[j] * self.cross_section_off[j]  # cis0_neutral absorption
                        K[:, 12, 11] = K[:, 12, 11] + PF[j] * self.cross_section_immature[j]  # cis0_immature absorption
                        K[:, 5, 4] = K[:, 5, 4] + PF[j] * self.cross_section_triplet[j]  # triplet absorption

            # Radiative & non-radiative decays (fluorescence or excited state relaxations)
            K[:, 0, 1] = PFeye / self.tau_on  # On state lifetime trans anionic (fluorescence emission)
            K[:, 2, 3] = PFeye / self.tau_on  # On state lifetime cis anionic
            K[:, 7, 8] = PFeye / self.tau_off  # Off state lifetime trans neutral
            K[:, 9, 10] = PFeye / self.tau_off  # Off state lifetime cis neutral
            K[:, 11, 12] = PFeye / self.tau_on  # On state lifetime cis immature (fluorescence emission)
            K[:, 4, 5] = PFeye / self.tau_triplet_excited  # Triplet excited state lifetime

            # Switching processes (transitions between states, also triplet)
            K[:, 2, 1] = PFeye * (self.qy_trans_to_cis_anionic / self.tau_on)  # Trans to cis anionic
            K[:, 0, 3] = PFeye * (self.qy_cis_to_trans_anionic / self.tau_on)  # Off switching
            K[:, 4, 3] = PFeye * (self.qy_isc / self.tau_on)  # Triplet state formation
            K[:, 3, 5] = PFeye * (
                    self.qy_re_isc / self.tau_triplet_excited)  # Light induced relaxation of triplet state
            K[:, 2, 4] = PFeye / self.tau_triplet  # Triplet state "thermal" relaxation
            K[:, 9, 8] = PFeye * (self.qy_trans_to_cis_neutral / self.tau_off)  # On switching
            K[:, 7, 10] = PFeye * (self.qy_cis_to_trans_neutral / self.tau_off)  # Cis to trans neutral
            K[:, 7, 12] = PFeye * (self.qy_off_im / self.tau_on)  # Cis immature to trans neutral

            # Bleaching processes
            K[:, 6, 1] = PFeye * (self.qy_bleaching_on / self.tau_on)  # Bleaching from trans anionic
            K[:, 6, 3] = PFeye * (self.qy_bleaching_on / self.tau_on)  # Bleaching from cis anionic
            K[:, 6, 4] = PFeye * (self.qy_bleaching_triplet / self.tau_triplet)  # Bleaching from triplet0
            K[:, 6, 5] = PFeye * (self.qy_bleaching_triplet_exc / self.tau_triplet_excited)  # Bleaching from triplet1
            K[:, 6, 8] = PFeye * (self.qy_bleaching_off / self.tau_off)  # Bleaching from trans neutral
            K[:, 6, 10] = PFeye * (self.qy_bleaching_off / self.tau_off)  # Bleaching from cis neutral
            K[:, 6, 12] = PFeye * (self.qy_bleaching_on / self.tau_on)  # Bleaching from cis immature

            # Protonation/deprotonation processes & protein re-arrangement
            K[:, 7, 0] = PFeye / self.tau_prot_trans  # Trans state protonation
            K[:, 11, 9] = PFeye / self.tau_maturation  # Cis neutral re-arrangement
            K[:, 11, 2] = PFeye / self.tau_prot_cis  # Cis state protonation
            K[:, 2, 11] = PFeye / self.tau_deprot_cis  # Cis state deprotonation
        ############################################################################################################

        # Compute diagonal. If the matrix is built properly, the diagonal terms are the negative sum of each column

        for i in range(PF.shape[1]):
            K_sum_columns = np.sum(K[i, :, :, ], axis=0)
            K_diagonal = np.diagflat(K_sum_columns)
            K[i, :, :] = K[i, :, :] - K_diagonal
        return K

    def initialCondition(self):
        t0_population = np.array(self.nstates)
        t0_population = self.initial_populations
        return t0_population


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Create the fluorophore model
    rsEGFP2_simple_model = NegativeSwitchers(extincion_coeff_on=[5260, 51560],
                                             extincion_coeff_off=[22000, 60],
                                             wavelength=[405, 488],
                                             lifetime_on=1.6E-6,
                                             lifetime_off=20E-9,
                                             qy_cis_to_trans_anionic=1.65E-2,
                                             qy_trans_to_cis_neutral=0.33,
                                             qy_fluorescence_on=0.35,
                                             initial_populations=[1, 0, 0, 0])
    # initial_populations: off_ground, off_excite, on_ground, on_excite

    # rsEGFP2_full_model = NegativeSwitchers(extincion_coeff_on=[5260, 51560, 0],
    #                                        extincion_coeff_off=[22000, 60, 0],
    #                                        wavelength=[405, 488, 592],
    #                                        lifetime_on=1.6E-6,
    #                                        lifetime_off=20E-9,
    #                                        qy_cis_to_trans_anionic=1.73E-2,
    #                                        qy_trans_to_cis_neutral=0.33,
    #                                        qy_cis_to_trans_neutral=0.23,
    #                                        qy_trans_to_cis_anionic=2e-3,
    #                                        qy_fluorescence_on=0.35,
    #                                        initial_populations=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                                        extincion_coeff_triplet=[2e3, 10e3, 7.5e3],
    #                                        lifetime_triplet=5,
    #                                        lifetime_triplet_excited=1E-9,
    #                                        inter_system_crossing=0,
    #                                        reverse_inter_system_crossing=1.2e-3,
    #                                        qy_bleaching_on=0,
    #                                        qy_bleaching_off=0,
    #                                        qy_bleaching_triplet=0,
    #                                        lifetime_deprot_cis=825e-3,
    #                                        lifetime_prot_trans=48e-3,
    #                                        pKa_cis=5.9,
    #                                        pH=7.5,
    #                                        qy_im_to_off=1.7e-2,
    #                                        qy_fluorescence_im=0.35,
    #                                        lifetime_maturation=5e-3,
    #                                        extincion_coeff_immature=[7e3, 21e3, 0],
    #                                        nspecies=13)

    off_switching_pulse = ModulatedLasers(wavelengths=[405, 488], power_densities=[0.5, 0.0], pulse_widths=[0.5, 0.0],
                                          t_start=[1, 2.5], dwell_time=30)

    off_switching_experiment = Experiment(illumination=off_switching_pulse,
                                          fluorophore=rsEGFP2_simple_model)

    populations = off_switching_experiment.solve_kinetics(0.01)

    print(populations[-1, 0])
    print(populations[-1, 1])
    print(populations[-1, 2])
    print(populations[-1, 3])

    plt.figure()
    plt.plot(populations[:, 0], label='off_ground')
    plt.plot(populations[:, 1], label='off_excite')
    plt.plot(populations[:, 2], label='on_ground')
    plt.plot(populations[:, 3], label='on_excite')
    plt.legend()
    plt.show()
