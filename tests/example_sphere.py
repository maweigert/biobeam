"""


mweigert@mpi-cbg.de

"""


import numpy as np

from biobeam import Bpm3d


if __name__ == '__main__':

    dx = .1
    x = dx*np.arange(-128,128)

    Z,Y,X = np.meshgrid(x,x,x, indexing="ij")
    R = np.sqrt(X**2+Y**2+Z**2)
    dn = .1*(R<3.)

    m = Bpm3d(dn = dn, units = (dx,)*3)

    u1 = m.propagate(u0 = m.u0_beam())

    
    #u2 = m.propagate(u0 = m.u0_beam(), dn_mean_method="local")
    #u = m.propagate(dn_mean_method="local")