"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
from biobeam import SimLSM_Cylindrical


def create_dn_and_signal(N = 512):
    """a refractive sphere and stripes signal"""
    x = np.linspace(-50,50,N)
    Xs = np.meshgrid(x,x,x,indexing = "ij")
    R = np.sqrt(np.sum([_X**2 for _X in Xs], axis = 0))
    dn = .04*(R<20)
    signal = (np.sin(2.*Xs[2])>0)*(np.sin(2.*Xs[1])>0)
    return dn, signal

dn, signal = create_dn_and_signal()


#create a microscope simulator
m = SimLSM_Cylindrical(dn = dn, signal = signal,
                       NA_illum= .1, NA_detect=.6,
                       n_volumes=2,
                       size = (100,100,100), n0 = 1.33)

psfs = m.psf_grid_z(cz=-20, grid_dim=(16,16), with_sheet = False)
