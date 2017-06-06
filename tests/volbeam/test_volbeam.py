"""


mweigert@mpi-cbg.de
"""

from __future__ import print_function, unicode_literals, absolute_import, division


import numpy as np
import os
from biobeam.beam_gui.volbeam import volbeam

def get_volume(N=256):
    from gputools import perlin3
    from tifffile import imread, imsave

    fname = "data/dn_%s.tif"%N
    if os.path.exists(fname):
        return imread(fname)



    print("creating volume...")

    _x = np.linspace(-1,1,N)
    R = np.sqrt(np.sum([_X**2 for _X in np.meshgrid(_x,_x,_x,indexing = "ij")],axis =0))

    dn = .05*(R<.2)
    dn += .01*perlin3((N,)*3, scale = 4)

    print("... done")
    imsave(fname, dn)

    return dn


if __name__=='__main__':
    dn = get_volume(512)

    volbeam(dn, size=(100,)*3, blocking = True)
