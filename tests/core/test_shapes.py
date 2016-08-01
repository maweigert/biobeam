"""

mweigert@mpi-cbg.de


"""
import numpy as np
from biobeam import Bpm3d


def get_dn():
    Nx = 256
    Ny = 256
    Nz = 256
    x = np.linspace(-Nx, Nx, Nx)/min((Nx, Ny, Nz))
    y = np.linspace(-Ny, Ny, Ny)/min((Nx, Ny, Nz))
    z = np.linspace(-Nz, Nz, Nz)/min((Nx, Ny, Nz))
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    R = np.sqrt(Z**2+Y**2+X**2)
    dn = .01*(R<.6)
    return dn


def test_shapes():
    dn1 = get_dn()
    p1, p2 = 30, 50
    dn2 = np.pad(dn1, ((0,)*2, (p1,)*2, (p2,)*2), mode="constant")

    m1 = Bpm3d(dn=dn1, units=(.1,)*3)
    u1 = m1.propagate(return_comp="intens")

    m2 = Bpm3d(dn=dn2, units=(.1,)*3)
    u2 = m2.propagate(return_comp="intens")[:, p1:-p1, p2:-p2]

    print "maximal absolute difference: ", np.amax(np.abs(u1-u2))
    return u1, u2


if __name__=='__main__':
    u1, u2 = test_shapes()
