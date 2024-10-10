""" Read temporal data in Par_Strat3d.hst and Par_Strat3d.phst """

import io
import numpy as np
import matplotlib.pyplot as plt
from ..plt import plt_params, ax_labeling


class ParticleHistory:
    """ Read particle data from Par_Strat3d.phst """

    def __init__(self, phst_filepath):
        """
        Read particle history dump for further analyses
        :param phst_filepath: the file path to Par_Start3d.phst
        """

        with open(phst_filepath) as phst_file:
            particle_history = phst_file.read().splitlines()
        particle_history = [x for x in particle_history if len(x) != 0 and x[0] != '#']
        par_global_scalars = '\n'.join(particle_history[::2])
        par_type_scalars = '\n'.join(particle_history[1::2])
        par_global_scalars = np.genfromtxt(io.BytesIO(par_global_scalars.encode()))
        par_type_scalars = np.genfromtxt(io.BytesIO(par_type_scalars.encode()))

        # mass, Mx/y/z, KEx/y/z are averaged by volume in the Particle-in-Mesh way
        # e.g., KEx = SUM_particles(0.5*rho_p*v1^2)/V, where rho_p is m_par/dVol, V=Lx*Ly*Lz, dVol=dx*dy*dz
        # here mass is M(total_par)/V(total)
        self.time, self.d_max, self.stiffmax, self.Edot, self.mass, \
            self.Mx, self.My, self.Mz, self.KEx, self.KEy, self.KEz = par_global_scalars.T
        self.time /= (2 * np.pi)

        # all the following values are computed directly from all the particles
        # e.g., Vx = SUM_particles(v1)/Npar
        self.Xavg, self.Yavg, self.Zavg, self.Vx, self.Vy, self.Vz, \
            self.Xvar, self.Yvar, self.Zvar, self.Vxvar, self.Vyvar, self.Vzvar = par_type_scalars.T

        # for more convenience
        self.M = np.asarray([self.Mx, self.My, self.Mz])
        self.KE = np.asarray([self.KEx, self.KEy, self.KEz])
        self.Ravg = np.asarray([self.Xavg, self.Yavg, self.Zavg])
        self.V = np.asarray([self.Vx, self.Vy, self.Vz])
        self.Rvar = np.asarray([self.Xvar, self.Yvar, self.Zvar])
        self.Vvar = np.asarray([self.Vxvar, self.Vyvar, self.Vzvar])

    def par_stats(self, z='z', etar=None, leg_loc='best', ax=None):
        """
        Plot the maximum density and scale height as a function of time for particles
        :param z: define the vertical direction (e.g., 'z' or 'y')
        :param etar: if not None, H_p will be plotted in units of eta r
        :param leg_loc: set the location of legend
        :param ax: Axes object for plotting; will create one if None
        """

        new_ax_flag = False
        if ax is None:
            plt_params("s")
            fig, ax = plt.subplots()
            new_ax_flag = True

        ax.semilogy(self.time*2*np.pi, self.d_max, 'r', lw=2, alpha=0.8, label=r"$\rho_{\rm p, max}[\rho_{\rm g,0}]$")
        ax_labeling(ax, x=r"$t[\Omega^{-1}]$", y=r"$\rho_{\rm p, max}[\rho_{\rm g,0}]$")

        ax2 = ax.twinx()
        if z == 'z':
            H_p = self.Zvar
        elif z == 'y':
            H_p = self.Yvar
        else:
            print("Warning: unknown vertical direction:", z, ". Using default Zvar.")
            H_p = self.Zvar
        if etar is not None: H_p /= etar
        H_p_label = r"$H_{\rm p}[H_{\rm g}]$" if etar is None else r"$H_{\rm p}[\eta r]$"
        ax2.plot(self.time*2*np.pi, H_p, 'b', lw=2, alpha=0.8, label=H_p_label)
        ax2.set_ylabel(H_p_label)
        ax.legend(ax.lines + ax2.lines, [l.get_label() for l in ax.lines + ax2.lines], loc=leg_loc)

        if new_ax_flag:
            return fig, ax


class MultiParticleHistory:
    """ Read particle data from Par_Strat3d.phst """

    def __init__(self, phst_filepath, num_species):
        """
        Read particle history dump for further analyses
        :param phst_filepath: the file path to Par_Start3d.phst
        """

        with open(phst_filepath) as phst_file:
            particle_history = phst_file.read().splitlines()
        particle_history = [x for x in particle_history if len(x) != 0 and x[0] != '#']
        par_global_scalars = '\n'.join(particle_history[::num_species+1])
        par_global_scalars = np.genfromtxt(io.BytesIO(par_global_scalars.encode()))
        par_type_scalars = []
        for idx in range(1, num_species+1):
            _par_type_scalars = '\n'.join(particle_history[idx::num_species+1])
            par_type_scalars.append( np.genfromtxt(io.BytesIO(_par_type_scalars.encode())) )
        par_type_scalars = np.asarray(par_type_scalars)

        # mass, Mx/y/z, KEx/y/z are averaged by volume in the Particle-in-Mesh way
        # e.g., KEx = SUM_particles(0.5*rho_p*v1^2)/V, where rho_p is m_par/dVol, V=Lx*Ly*Lz, dVol=dx*dy*dz
        # here mass is M(total_par)/V(total)
        self.time, self.d_max, self.stiffmax, self.Edot, self.mass, \
            self.Mx, self.My, self.Mz, self.KEx, self.KEy, self.KEz = par_global_scalars.T
        self.time /= (2 * np.pi)

        # all the following values are computed directly from all the particles
        # e.g., Vx = SUM_particles(v1)/Npar
        self.Xavg, self.Yavg, self.Zavg, self.Vx, self.Vy, self.Vz, \
            self.Xvar, self.Yvar, self.Zvar, self.Vxvar, self.Vyvar, self.Vzvar = \
            np.swapaxes(np.swapaxes(par_type_scalars, 0, 2), 1, 2)

        # for more convenience
        self.M = np.asarray([self.Mx, self.My, self.Mz])
        self.KE = np.asarray([self.KEx, self.KEy, self.KEz])
        self.Ravg = np.swapaxes(np.asarray([self.Xavg, self.Yavg, self.Zavg]), 0, 1)
        self.V = np.swapaxes(np.asarray([self.Vx, self.Vy, self.Vz]), 0, 1)
        self.Rvar = np.swapaxes(np.asarray([self.Xvar, self.Yvar, self.Zvar]), 0, 1)
        self.Vvar = np.swapaxes(np.asarray([self.Vxvar, self.Vyvar, self.Vzvar]), 0, 1)


class GasHistory:
    """ Read gas data from Par_Start3d.hst"""

    def __init__(self, hst_filepath):
        """
        Read gas history dump for further analyses
        :param hst_filepath: the file path for Par_Start3d.hst
        """

        gas_hst_scalars = np.loadtxt(hst_filepath)

        # all these values are averaged by volume
        # e.g., KEx = [SUM_cells(dVol*0.5*M1^2/rho_g)]/V, where dVol=dx*dy*dz, V=Lx*Ly*Lz.
        # here mass isn't total mass, is also volume-averaged value, in other word, <rho_g> = M_tot/V
        self.time, self.dt, self.mass, self.Mx, self.My, self.Mz, \
            self.KEx, self.KEy, self.KEz, self.RhoVxdVy = gas_hst_scalars.T
        self.time /= (2 * np.pi)

        # for more convenience
        self.M = np.asarray([self.Mx, self.My, self.Mz])
        self.KE = np.asarray([self.KEx, self.KEy, self.KEz])


class LogHistory:
    """ Read output log from output.txt """

    def __init__(self, log_filepath, SMR=False):
        """
        Read log file for further analyses
        :param log_filepath: the file path for (usually named) output.txt
        :param SMR: if SMR is used in Athena
        N.B.: both the replenish_ratio and mass_loss_rate are for the gas within the box
        """

        self.time, self.dt, self.rt, self.replenish_ratio, \
            self.mass_loss_rate = self._read_in_one_log(log_filepath, SMR=SMR)

    def append_log(self, log_filepath, SMR=False):
        """ Append more info from a new log file """

        tmp_time, tmp_dt, tmp_rt, tmp_replenish_ratio, tmp_mass_loss_rate = self._read_in_one_log(log_filepath, SMR=SMR)

        if tmp_time[-1] < self.time[-1]:
            print("WARNING: the appended log seems to have an earlier end-time than the original one",
                  tmp_time[-1], self.time[-1])

        if tmp_time[0] < self.time[0]:
            raise RuntimeError("The appended log seems to have an earlier start-time than the original one",
                               tmp_time[0], self.time[0], "It is recommended to read in the earlier log first.")

        self.dt = np.hstack([self.dt[self.time < tmp_time[0]], tmp_dt])
        self.rt = np.hstack([self.rt[self.time < tmp_time[0]], tmp_rt])
        self.time = np.hstack([self.time[self.time < tmp_time[0]], tmp_time])
        if self.replenish_ratio.size > 0:
            self.replenish_ratio = np.hstack([self.replenish_ratio[self.time < tmp_time[0]], tmp_replenish_ratio])
            self.mass_loss_rate = np.hstack([self.mass_loss_rate[self.time < tmp_time[0]], tmp_mass_loss_rate])

    def _read_in_one_log(self, log_filepath, SMR=False):
        """
        Read log file for further analyses
        :param log_filepath: the file path for (usually named) output.txt
        :param SMR: if SMR is used in Athena
        N.B.: both the replenish_ratio and mass_loss_rate are for the gas within the box
        """

        with open(log_filepath) as f:
            log_output = f.read().splitlines()

        # first, get time and timestep
        log_output = [line for line in log_output if len(line) != 0]
        log_data = [line for line in log_output if line[:5] == "cycle"]
        time = np.zeros(len(log_data))
        dt = np.zeros(len(log_data))
        rt = np.zeros(len(log_data))
        for i, line in enumerate(log_data):
            time_pos = line.find("time")
            time[i] = float(line[time_pos + 5:time_pos + 17])
            dt_pos = line.find("dt")
            dt[i] = float(line[dt_pos + 3:dt_pos + 15])
            rt_pos = line.find("rt")
            if rt_pos != -1:
                rt[i] = float(line[rt_pos + 3:rt_pos + 15])

        # then, get mass replenishment if presented
        replenish_ratio, mass_loss_rate = np.array([]), np.array([])
        if SMR is False:
            log_data = [line for line in log_output if line[:8] == "mratio ="]
        else:
            log_data = [line for line in log_output if line[:11] == "mratio[1] ="]

        if len(log_data) > 0:
            replenish_ratio = np.zeros(len(log_data))
            for i, line in enumerate(log_data):
                replenish_ratio[i] = float(line[12:34]) if SMR else float(line[9:])
            if replenish_ratio.size != time.size:
                # Usually in the end Athena will print out cycle one more time
                if time.size - replenish_ratio.size == 1:
                    time = time[:-1]
                    dt = dt[:-1]
                    rt = rt[:-1]
                else:
                    print("WARNING: the length of data (replenish_ratio vs. time) doesn't match.")
            mass_loss_rate = (1.0 - 1.0/replenish_ratio) / dt

        return time, dt, rt, replenish_ratio, mass_loss_rate