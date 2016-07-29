"""the main method for beam propagation in media with coated spheres"""

import numpy as np
import numpy.testing as npt
from time import time
from biobeam import Bpm3d


def test_vols(vols = range(1,6)):


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
        print "n_volume = %s \t diff = %.4f \t time = %.4fs"%(nvols, np.mean(np.abs(u0-u)),t)



if __name__ == '__main__':

    test_vols()


    #plot_some()

