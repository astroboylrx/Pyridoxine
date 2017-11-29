""" Read the simulation data on Dusty Disk in 1D """

from array import array
import struct
import numpy as np
import scipy.integrate as spint
import astropy.units as u
import astropy.constants as c


class DustyDiskData:
    """ Store surface densities by recording data positions in binary file """

    def __init__(self, filepath, num_grids):
        """
        Initialize data sequence and record file path
        :param filepath:
        :param num_grids:
        """
        self.pos = []
        self.filepath = filepath
        self.f = open(self.filepath, 'rb')
        self.ng = num_grids
        self.f.close()

    def __getitem__(self, index):
        """
        Overload [.] operator to obtain data at a specific time
        :param index: index in terms of evolution (time)
        :return: return surface density along radius as numpy ndarray
        """
        self.f = open(self.filepath, 'rb')
        self.f.seek(self.pos[index])
        tmp_arr = array('d')
        tmp_arr.fromfile(self.f, self.ng)
        self.f.close()
        return np.asarray(tmp_arr)


class DustyDisk:
    """ Read surface density data from Dusty Disk simulation in 1D """

    def __init__(self, filepath, num_species, sizes, r_in=0.1, r_out=400., r_c=10.0, bump_peak=0.1, bump_width=0.025):
        """
        Read surface density data, construct data grid and
         calculate all relevant physical quantities
        :param filepath: the file path to dd1d result
        :param num_species: the # of dust species
        :param sizes: particle sizes (must has "num_species" elements)
        :param r_in: the inner radius in simulation
        :param r_out: the outer radius in simulation
        :param r_c: bump position
        :param bump_peak: the peak value of the dust bump (in terms of Z)
        :param bump_width: the width of the dust bump (in units of r_c)
        """

        toau = u.au.to(u.meter)
        toyr = u.second.to(u.year)

        # open binary data file
        f = open(filepath, 'rb')
        tmp_ng = f.read(4)  # unsigned int
        self.ng = struct.unpack('I', tmp_ng)[0]
        print("num_grids = ", self.ng)

        self.r_AU = np.linspace(np.sqrt(r_in), np.sqrt(r_out), self.ng) ** 2
        self.r = np.linspace(np.sqrt(r_in * toau), np.sqrt(r_out * toau), self.ng) ** 2
        self.s = 2 * np.sqrt(self.r)  # uniform grid
        self.ds = 2 * (np.sqrt(self.r[-1]) - np.sqrt(self.r[0])) / (self.ng - 1)
        self.two_ds = 2 * self.ds
        self.sqrt2pi = np.sqrt(2 * np.pi)
        self.bump_peak = bump_peak
        self.bump_width = bump_width * r_c
        self.num_spec = num_species
        if self.num_spec > 1:
            raise ValueError("Haven't implemented for multi-species dust components.")
        self.sizes = np.asarray(sizes)
        self.mu = 2.3  # mean molecular weight
        self.sigma_H2 = 2e-19  # m^2, cross section
        self.rho_s = 2000  # kg / m^3

        # primary data to read: time, Sigma_gas, Sigma_p
        self.Sigma_gas = DustyDiskData(filepath, self.ng)
        self.Sigma_p = DustyDiskData(filepath, self.ng)
        self.time = []

        eof = f.seek(0, 2)
        f.seek(4)
        itemsize = array('d').itemsize
        print("itemsize=", itemsize)
        while f.tell() != eof:
            self.time.append(struct.unpack('d', f.read(itemsize))[0])
            f.seek(itemsize, 1)  # ratio
            self.Sigma_gas.pos.append(f.tell())
            f.seek(itemsize * self.ng, 1)
            self.Sigma_p.pos.append(f.tell())
            f.seek(itemsize * self.ng, 1)
            # print(f.tell())
        self.time = np.asarray(self.time) * toyr

        # values in the midplane of r_c
        self.r_c = r_c * toau
        self.Omega_K_rc = np.sqrt(c.M_sun.value * c.G.value / self.r_c ** 3)
        self.Mdot_gas = 5e-10
        self.alpha_rc = 0.0001
        self.H_over_r_rc = 0.0592
        self.H_gas_rc = self.r_c * self.H_over_r_rc
        self.c_s_rc = self.H_gas_rc * self.Omega_K_rc
        self.nu_rc = self.alpha_rc * self.c_s_rc ** 2 / self.Omega_K_rc
        # self.t_visc = (self.bump_width*self.r_c)**2 / self.nu_rc

        # values in the midplane along r
        self.alpha = np.zeros(self.ng) + self.alpha_rc  # may have a profile in the future
        self.H_gas = self.H_gas_rc * (self.r / self.r_c) ** (1.25)
        self.Omega_K = self.Omega_K_rc * (self.r / self.r_c) ** (-3 / 2)
        self.c_s = self.H_gas * self.Omega_K
        self.c_s2 = self.c_s ** 2
        self.nu = self.alpha * self.c_s2 / self.Omega_K
        self.v_K = self.Omega_K * self.r

        # establish the vertical numerical grids
        self.num_z_grids = self.ng // 2
        self.z = np.array([np.linspace(-5 * hg, 5 * hg, self.num_z_grids) for hg in
                           self.H_gas]).transpose()  # in order to fit the broadcast rules
        self.z2_over_2 = self.z ** 2 / 2

        # numerical factors to save computation time
        self.efactor_gas = np.exp(-self.z ** 2 / 2.0 / self.H_gas ** 2)
        self.c_s_rz = np.repeat([self.c_s], self.num_z_grids, axis=0)
        self.Omega_K_rz = np.repeat([self.Omega_K], self.num_z_grids, axis=0)

        # initialize variables as zeros
        self.__rho_gas0 = np.zeros(self.ng)  # 0 means midplane value
        self.__rho_gas = np.zeros([self.num_z_grids, self.ng])
        self.__lambda_gas0 = np.zeros(self.ng)  # 0 means midplane value
        self.__lambda_gas = np.zeros([self.num_z_grids, self.ng])
        self.__P_gas = np.zeros(self.ng)
        self.__P_gradient = np.zeros(self.ng)
        self.__eta = np.zeros(self.ng)
        self.__etav_K = np.zeros(self.ng)
        self.__mdf_NSH_vgr = np.zeros([self.num_z_grids, self.ng])
        self.__v_gas_visc = np.zeros(self.ng)
        self.__V_gasr = np.zeros(self.ng)

        self.__rho_p0 = np.zeros([self.num_spec, self.ng])  # 0 means midplane value
        self.__rho_p = np.zeros([self.num_spec, self.num_z_grids, self.ng])
        self.__H_p = np.zeros([self.num_spec, self.ng])
        self.__tau_s0 = np.zeros([self.num_spec, self.ng])
        self.__tau_s = np.zeros([self.num_spec, self.num_z_grids, self.ng])
        self.__epsilon0 = np.zeros([self.num_spec, self.ng])
        self.__epsilon = np.zeros([self.num_spec, self.num_z_grids, self.ng])
        self.__mdf_NSH_vpr = np.zeros([self.num_spec, self.num_z_grids, self.ng])
        self.__V_pr = np.zeros([self.num_spec, self.ng])

        self.__velocity_state = -1
        self.V_gasr(0)
        self.rc_index = np.argmin(abs(self.r_c - self.r))
        self.t_drift = np.abs(self.r[self.rc_index] / self.__V_pr[0][self.rc_index] * toyr)
        print("t_drift = ", self.t_drift, "yrs")

    def __process_index(self, i):
        """
        Handle negative index value
        :param i: index in terms of evolution (time)
        :return: original i if positive, re-mapped i if negative
        """

        if i <= -1:
            return self.time.size + i
        else:
            return i

    def P_gas(self, i):
        """
        Calculate gas pressure in the disk midplane
        :param i: index in terms of evolution (time)
        :return: gas pressure in the disk midplane at time[i]
        """

        i = self.__process_index(i)
        self.__P_gas = self.Sigma_gas[i] * self.c_s * self.Omega_K / np.sqrt(2 * np.pi)
        return self.__P_gas

    def P_gradient(self, i):
        """
        Calculate gas pressure gradient in the disk midplane
        :param i: index in terms of evolution (time)
        :return: gas pressure gradient in the disk midplane at time[i]
        """

        i = self.__process_index(i)
        self.__P_gradient = 2 / self.s * self.partial_derivative_s(self.P_gas(i))
        return self.__P_gradient

    def rho_gas0(self, i):
        """
        Calculate gas density in the disk midplane
        :param i: index in terms of evolution (time)
        :return: gas density in the disk midplane at time[i]
        """

        i = self.__process_index(i)
        self.__rho_gas0 = self.Sigma_gas[i] / self.H_gas / np.sqrt(2 * np.pi)
        return self.__rho_gas0

    def rho_gas(self, i):
        i = self.__process_index(i)
        self.__rho_gas = self.rho_gas0(i) * self.efactor_gas
        return self.__rho_gas

    def eta(self, i):
        i = self.__process_index(i)
        self.__eta.fill(0.)
        tmp_rho_gas0 = self.rho_gas0(i)
        positive_i = tmp_rho_gas0 > 0
        self.__eta[positive_i] = \
            -1 / (2 * tmp_rho_gas0[positive_i] * self.Omega_K[positive_i] ** 2 * self.r[positive_i]) * \
            self.P_gradient(i)[positive_i]
        return self.__eta

    def calculate_spatial_density(self, i):
        self.rho_gas(i)
        # remember there may be 0 in rho_gas

        tmp_factor = self.mu * c.m_p.value / self.sigma_H2
        self.__lambda_gas0.fill(0.)
        self.__lambda_gas.fill(0.)
        positive_gas_indices = self.__rho_gas0 > 0
        self.__lambda_gas0[positive_gas_indices] = tmp_factor / self.__rho_gas0[positive_gas_indices] * 9 / 4
        positive_rho_gas = self.__rho_gas > 0
        self.__lambda_gas[positive_rho_gas] = tmp_factor / self.__rho_gas[positive_rho_gas] * 9 / 4

        self.__tau_s0.fill(0.)
        self.__tau_s.fill(0.)
        for j in range(self.num_spec):
            # self.__lambda_gas0 is already 0 at rho_gas0 = 0
            Epstein_indices = self.sizes[j] < self.__lambda_gas0
            self.__tau_s0[j][Epstein_indices] = self.Omega_K[Epstein_indices] * self.rho_s * self.sizes[j] / \
                                                self.__rho_gas0[Epstein_indices] / self.c_s[Epstein_indices]

            Stokes_indices = ((self.sizes[j] > self.__lambda_gas0) & (self.__rho_gas0 > 0))
            # lambda_gas0 already contains 9/4
            self.__tau_s0[j][Stokes_indices] = self.Omega_K[Stokes_indices] * self.rho_s * self.sizes[j] ** 2 \
                                               / (self.__rho_gas0[Stokes_indices] * self.c_s[Stokes_indices] *
                                                  self.__lambda_gas0[Stokes_indices])

            # self.lambda_gas is already 0 at rho_gas = 0
            Epstein_indices = self.sizes[j] < self.__lambda_gas
            self.__tau_s[j][Epstein_indices] = self.Omega_K_rz[Epstein_indices] * self.rho_s * self.sizes[j] / \
                                               self.__rho_gas[Epstein_indices] / self.c_s_rz[Epstein_indices]

            Stokes_indices = ((self.sizes[j] > self.__lambda_gas) & (self.__rho_gas > 0))
            # lambda_gas already contains 9/4
            self.__tau_s[j][Stokes_indices] = self.Omega_K_rz[Stokes_indices] * self.rho_s * self.sizes[j] ** 2 \
                                              / (self.__rho_gas[Stokes_indices] * self.c_s_rz[Stokes_indices] *
                                                 self.__lambda_gas[Stokes_indices])

        self.__H_p = self.H_gas * np.sqrt(self.alpha / (self.alpha + self.__tau_s0))
        self.__rho_p0 = self.Sigma_p[i] / self.sqrt2pi / self.__H_p

        for j in range(self.num_spec):
            self.__epsilon0[j][positive_gas_indices] = self.__rho_p0[j][positive_gas_indices] / self.__rho_gas0[
                positive_gas_indices]
            self.__rho_p[j] = self.__rho_p0[j] * np.exp(-self.z2_over_2 / self.__H_p[j] ** 2)
            self.__epsilon[j][positive_rho_gas] = self.__rho_p[j][positive_rho_gas] / self.__rho_gas[positive_rho_gas]
        self.__etav_K = self.eta(i) * self.v_K
        # skip calculating D_p

    def density_weighted_vr(self, i):
        """ Calculate vertically-density-weighted NSH radial velocity """

        self.calculate_spatial_density(i)
        self.__v_gas_visc.fill(0.)  # self.calculate_v_gas_visc(i)

        if self.num_spec == 1:
            tmp_tau_s2 = self.__tau_s ** 2
            tmp_epsilon1 = 1 + self.__epsilon
            # tmp_numerator = tmp_tau_s2 + tmp_epsilon1
            tmp_denominator = tmp_tau_s2 + tmp_epsilon1 ** 2

            # this ensures vpr = 0 at rho_gas = 0
            # self.__mdf_NSH_vpr = (-2*self.__tau_s*self.__etav_K + tmp_epsilon1*self.__v_gas_visc) / tmp_denominator
            # self.__mdf_NSH_vgr = ((2*self.__epsilon*self.__tau_s*self.__etav_K + tmp_numerator*self.__v_gas_visc) / tmp_denominator)[0]
            self.__mdf_NSH_vpr = -2 * self.__tau_s * self.__etav_K / tmp_denominator
            self.__mdf_NSH_vgr = (self.__epsilon * -self.__mdf_NSH_vpr)[0]

        elif self.num_spec > 1:
            raise ValueError("Haven't implemented for multi-species dust components.")

        self.__V_gasr.fill(0.)
        tmp_indices = self.Sigma_gas[i] > 0
        self.__V_gasr[tmp_indices] = spint.simps(self.__rho_gas * self.__mdf_NSH_vgr, self.z, axis=0)[tmp_indices] / \
                                     self.Sigma_gas[i][tmp_indices]

        self.__V_pr.fill(0.)
        for j in range(self.num_spec):
            tmp_indices = self.Sigma_p[i] > 0
            self.__V_pr[j][tmp_indices] = spint.simps(self.__rho_p[j] * self.__mdf_NSH_vpr[j], self.z, axis=0)[
                                              tmp_indices] / self.Sigma_p[i][tmp_indices]
        self.__velocity_state = i

    def V_gasr(self, i):
        if self.__velocity_state != i:
            i = self.__process_index(i)
            self.density_weighted_vr(i)
        return self.__V_gasr

    def V_pr(self, i):
        if self.__velocity_state != i:
            i = self.__process_index(i)
            self.density_weighted_vr(i)
        return self.__V_pr[0]

    def tau_s0(self, i):
        if self.__velocity_state != i:
            i = self.__process_index(i)
            self.density_weighted_vr(i)
        return self.__tau_s0

    def tau_s(self, i):
        if self.__velocity_state != i:
            i = self.__process_index(i)
            self.density_weighted_vr(i)
        return self.__tau_s0

    def epsilon0(self, i):
        if self.__velocity_state != i:
            i = self.__process_index(i)
            self.density_weighted_vr(i)
        return self.__epsilon0

    def epsilon(self, i):
        if self.__velocity_state != i:
            i = self.__process_index(i)
            self.density_weighted_vr(i)
        return self.__epsilon

    def M_gas_flux(self, i):
        return self.V_gasr(i) * self.r * self.Sigma_gas[i] * 2 * np.pi

    def partial_derivative_s(self, profile, extrapolate=True):

        profile_shift_left = np.roll(profile, -1)
        profile_shift_right = np.roll(profile, 1)
        partial_profile_partial_s = (profile_shift_left - profile_shift_right) / self.two_ds

        if extrapolate:  # one-side differencing
            partial_profile_partial_s[0] = (- profile[2] + 4 * profile[1] - 3 * profile[0]) / self.two_ds
            partial_profile_partial_s[-1] = (3 * profile[-1] - 4 * profile[-2] + profile[-3]) / self.two_ds
        else:
            partial_profile_partial_s[0] = (profile[1] - profile[0]) / self.ds
            partial_profile_partial_s[-1] = (profile[-1] - profile[-2]) / self.ds

        return partial_profile_partial_s
