"""


mweigert@mpi-cbg.de

"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from biobeam import Bpm3d
from six.moves import zip
import matplotlib.pyplot as plt

def test_plane():
    dx = .02
    lam = .5
    Nx = 128
    Ny = 256
    Nz = 400
    dn0 = .1

    dn = dn0 * np.ones((Nz, Ny, Nx))

    m = Bpm3d(dn=dn,
              n0=1.,
              units=(dx,) * 3, lam=lam)

    # plane wave
    z = np.arange(Nz)
    u0 = np.exp(2.j * np.pi / lam * (1. + dn0) * z * dx)

    modes = ["none", "global", "local"]

    us = [m.propagate(dn_mean_method=mode) for mode in modes]

    for mode, u in zip(modes, us):
        print("diff (%s):\t%.3g" % (mode, np.mean(np.abs(u[:, Ny // 2, Nx // 2] - u0))))


if __name__ == '__main__':

    dx = .02
    lam = .5
    Nx = 128
    Ny = 256
    Nz = 400
    dn0 = .4

    dn = np.zeros((Nz,Ny,Nx))
    dn[Nz//3:2*Nz//3,Ny//3:2*Ny//3,Nx//3:2*Nx//3] = dn0

    m = Bpm3d(dn = dn,
            n0 =1.,
              units = (dx,)*3,lam = lam)

    modes = ["none", "global", "local"]

    us = [m.propagate(dn_mean_method=mode) for mode in modes]

    plt.figure(1)
    plt.clf()
    for _u,t in zip(us,modes):
        plt.plot(_u[:,Ny//2, Nx//2], label = t)

    plt.legend()
    plt.show()


