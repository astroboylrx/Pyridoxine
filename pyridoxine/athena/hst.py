""" Read temporal data in Par_Strat3d.hst and Par_Strat3d.phst """

import io
import numpy as np


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