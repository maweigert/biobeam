"""


mweigert@mpi-cbg.de

"""
import numpy as np
from biobeam import SimLSM_Cylindrical
import numpy.testing as npt

if __name__ == '__main__':

    Nx = 512
    Nz  = 300
    dx = .5

    dn = np.zeros((300,512,512))
    Z,Y,X = np.meshgrid(dx * np.arange(Nz),dx * (np.arange(Nx)-Nx/2),dx * (np.arange(Nx)-Nx/2), indexing = "ij")

    R = np.sqrt(X**2+Y**2+Z**2)

    dn = .06*(R<(dx*Nx/4))

    # dn = np.zeros((Nz,Nx,Nx))

    m = SimLSM_Cylindrical(dn = dn,
                           NA_detect=.8,
                        units = (dx,)*3,
                        n_volumes=1,
                        simul_xy_detect=(1024,1024),
                         )

    modes = ("none", "global", "local")

    hs = tuple(m.psf_grid_z((-Nz/2+20)*dx,grid_dim=(16,16),dn_mean_method = mode, zslice=16) for mode in modes)
    #hs = tuple(m.psf(((-Nz / 2 + 20) * dx,0,0),dn_mean_method=mode, zslice=16) for mode in modes)

    npt.assert_allclose(hs[0], hs[1], verbose=True)
    npt.assert_allclose(hs[0], hs[2], verbose=True)