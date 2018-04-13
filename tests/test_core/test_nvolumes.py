"""the main method for beam propagation in media with coated spheres"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import numpy.testing as npt
from time import time
from biobeam import Bpm3d
from six.moves import range


def test_speed(vols = list(range(1,6))):


    Nx = 256
    Ny = 256
    Nz = 256

    dn = np.random.uniform(0,.1,(Nz,Ny,Nx))

    us = []

    m = Bpm3d(dn = dn, units = (.1,)*3, n_volumes=1)
    u0 = m.propagate()

    for nvols in vols:
        m = Bpm3d(dn = dn, units = (.1, )*3, n_volumes=nvols)
        t = time()
        u = m.propagate()
        t = time()-t
        print("n_volume = %s \t diff = %.4f \t time = %.4fs"%(nvols, np.mean(np.abs(u0-u)),t))

def test_prop():
    Nx = 256
    Ny = 512
    Nz = 256

    x = np.linspace(-Nx,Nx,Nx)/min((Nx,Ny,Nz))
    y = np.linspace(-Ny,Ny,Ny)/min((Nx,Ny,Nz))
    z = np.linspace(-Nz,Nz,Nz)/min((Nx,Ny,Nz))

    Z,Y,X  = np.meshgrid(z,y,x, indexing = "ij")
    R = np.sqrt(Z**2+Y**2+X**2)

    dn = .1*(R<.8)

    m1 = Bpm3d(dn = dn, units = (.1, )*3, n_volumes=1)
    u1 = m1.propagate(return_comp="intens")
    m2 = Bpm3d(dn = dn, units = (.1, )*3, n_volumes=2)
    u2 = m1.propagate(return_comp="intens")


    assert np.allclose(u1,u2)
    return u1,u2



if __name__ == '__main__':

    #test_speed()

    u1, u2 = test_prop()


    #plot_some()

