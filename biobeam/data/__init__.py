"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division

import os
from scipy.misc import imread

datadir = os.path.abspath(os.path.dirname(__file__))


def tiling(N = 512):
    return imread(os.path.join(datadir,"biobeam-tiling_%s.png"%N), mode = "L")

