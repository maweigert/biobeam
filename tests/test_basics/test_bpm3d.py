"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
from biobeam import Bpm3d


def test_bpm3d():
    dn = np.zeros((128,)*3)
    dn[64:] = .1

    m = Bpm3d(dn = dn, units = (.1,)*3, lam = .5)

    u = m.propagate(m.u0_beam(NA = .5))