"""


mweigert@mpi-cbg.de

"""
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from biobeam.simlsm.simlsm import SimLSM_Base
from six.moves import range


class SimLSM_Lattice(SimLSM_Base):
    def __init__(self, dn=None,
                 signal=None,
                 shape=None,
                 size=None,
                 units=None,
                 lam_illum=.5,
                 NA_illum1=.4,
                 NA_illum2=.45,
                 sigma=0.1,
                 kpoints=6,
                 lam_detect=.5,
                 NA_detect=.7,
                 n0=1.33,
                 n_volumes=1,
                 zfoc_illum=None,
                 simul_xy_illum=None,
                 simul_z_illum=1,
                 simul_xy_detect=None,
                 simul_z_detect=1):

        self.dn = dn
        self.signal = signal

        dn_trans = dn.transpose(self.perm_illum).copy() if dn else None

        self._bpm_illum = Bpm3d(size=size,
                                shape=shape,
                                dn=dn_trans,
                                units=units,
                                lam=lam_illum,
                                simul_xy=simul_xy_illum,
                                simul_z=simul_z_illum,
                                n_volumes=n_volumes,
                                n0=n0)

        self._bpm_detect = Bpm3d(size=size,
                                 shape=shape,
                                 dn=dn,
                                 units=units,
                                 lam=lam_detect,
                                 simul_xy=simul_xy_detect,
                                 simul_z=simul_z_detect,
                                 n_volumes=n_volumes,
                                 n0=n0)

        self.NA_illum1 = NA_illum1
        self.NA_illum2 = NA_illum2
        self.kpoints = kpoints
        self.sigma = sigma
        self.NA_detect = NA_detect
        self.zfoc_illum = zfoc_illum
        self.dn = self._bpm_detect.dn
        self.units = self._bpm_detect.units
        self.Nx, self.Ny, self.Nz = self._bpm_detect.shape
        self.size = self._bpm_detect.size

        self._prepare_u0_all()

    def _prepare_u0_illum(self, zfoc):
        self.u0_illum = self._bpm_illum.u0_lattice(NA1=self.NA_illum1, zfoc=zfoc)

    def propagate_illum_single(self, cz=0, **bpm_kwargs):
        bpm_kwargs.update({"return_comp": "intens"})
        u0 = np.roll(self.u0_illum, int(cz/self._bpm_illum.dx), axis=0)
        u = self._bpm_illum.propagate(u0, **bpm_kwargs)
        return u.transpose(self.perm_illum_inv)

    def propagate_illum(self, cz=0, dx_parallel=None, **bpm_kwargs):
        bpm = self._bpm_illum
        bpm_kwargs.update({"return_comp": "intens"})
        u0 = np.roll(self.u0_illum, int(cz/self._bpm_illum.dx), axis=0)

        # prepare the parallel scheme
        max_NA = self.NA_illum if np.isscalar(self.NA_illum) else max(self.NA_illum)

        if dx_parallel is None:
            dx_parallel = 2*bpm.lam/max_NA

        print("dslm prop with parallelize beams of distance dx = %s mu"%dx_parallel)

        # the beamlet centers in the simul_xy coordinates
        ind_step = int(np.ceil(1.*dx_parallel/bpm.dx))
        # make sure its divisible by the grid size dimension
        ind_step = [i for i in range(ind_step, bpm.simul_xy[0]+1) if bpm.simul_xy[0]%i==0][0]

        if ind_step<=0:
            raise ValueError("dx resolution too coarse to propagate in parallel pleae increase dx_parallel")

        inds = np.arange(0, bpm.simul_xy[1], ind_step)

        print(inds)

        u0 = np.sum([np.roll(self.u0_illum, i, axis=0) for i in inds], axis=0)

        u = None
        # now scan
        for i in range(ind_step):
            print("propagating beamlets %s/%s"%(i+1, ind_step))
            u_part = self._bpm_illum.propagate(u0, **bpm_kwargs)
            u = u_part if u is None else u+u_part
            u0 = np.roll(u0, 1, axis=0)

        return u.transpose(self.perm_illum_inv)
