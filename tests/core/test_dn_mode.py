"""


mweigert@mpi-cbg.de

"""

import numpy as np
from biobeam import Bpm3d


if __name__ == '__main__':

    dx = .1
    lam = .5
    N = 8
    dn0 = .1

    dn = dn0*np.ones((2,N,N))

    m = Bpm3d(dn = dn,
            n0 =1.,
              units = (dx,)*3,lam = lam)

    #plane wave
    z = np.arange(dn.shape[0])
    u0 = np.exp(2.j*np.pi/lam*(1.+dn0)*z*dx)

    modes = ["none", "global","local"]

    us = [m.propagate(dn_mean_method=mode) for mode in modes]

    for mode,u in zip(modes,us):
        print "diff (%s):\t%.3g"%(mode,np.mean(np.abs(u[:,N/2,N/2]-u0)))




