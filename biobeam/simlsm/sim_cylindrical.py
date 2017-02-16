"""


mweigert@mpi-cbg.de

"""
from __future__ import absolute_import
import numpy as np
from biobeam.simlsm.simlsm import SimLSM_Base
from six.moves import range


class SimLSM_Cylindrical(SimLSM_Base):
    def _prepare_u0_illum(self, zfoc=None):
        self.u0_illum = self._bpm_illum.u0_cylindrical(NA=self.NA_illum, zfoc=zfoc)


if __name__=='__main__':

    dn = np.zeros((256, 512, 256))

    signal = np.zeros_like(dn)

    # some point sources
    np.random.seed(0)
    for _ in range(4000):
        k, j, i = np.random.randint(dn.shape[0]), np.random.randint(dn.shape[1]), np.random.randint(dn.shape[2])
        signal[k, j, i] = 1.

    if not "m" in locals():
        m = SimLSM_Cylindrical(dn=dn,
                               signal=signal,
                               NA_illum=.4,
                               NA_detect=.7,
                               units=(.1,)*3,
                               n0=1.33,
                               # simul_xy_detect=(512,512),
                               # simul_xy_illum=(512,512),
                               )

    # u1 = m.propagate_illum(cz=-30)
    # u2 = m.propagate_illum(cz=30)

    h = m.psf((10.,0,0))
    # from time import time
    # t = time()
    # hs = m.psf_grid_z(10,grid_dim = (8,16), zslice = 16)
    # print time()-t