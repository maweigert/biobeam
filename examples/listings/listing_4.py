"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
from biobeam import SimLSM_Cylindrical
from biobeam.data import tiling
from gputools import perlin3



def create_dn_and_signal(N = 256, offset = 0):
    """a refractive sphere and stripes signal"""
    x = np.linspace(-50,50,N)
    Xs = np.meshgrid(x,x,x,indexing = "ij")
    R = np.sqrt(np.sum([_X**2 for _X in Xs], axis = 0))
    dn = .07*(R<35)
    dn += 0.02*perlin3((N,N,N), scale =3)*(R<35)
    signal = np.einsum("i,jk",np.ones(N), tiling(N))
    return dn, signal


dn, signal = create_dn_and_signal(offset = -51)


#create a microscope simulator
m = SimLSM_Cylindrical(dn = dn, signal = signal,
                       NA_illum= .1, NA_detect=.45,
                       n_volumes=2,
                       size = (100,100,100), n0 = 1.33)

image = m.simulate_image_z(cz=-20, psf_grid_dim=(16,16), conv_sub_blocks=(2,2))[16]
