"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division

from biobeam import focus_field_beam, focus_field_cylindrical, focus_field_lattice



def test_gaussian():
    u = focus_field_beam((128,)*3,(.1,)*3,lam = .5, n0 = 1.33, NA = .7)

def test_bessel():
    u = focus_field_beam((128,) * 3, (.1,) * 3, lam=.5, n0=1.33, NA=(.7,.75))

def test_cylindrical():
    u = focus_field_cylindrical((128,) * 3, (.1,) * 3, lam=.5, n0=1.33, NA=.7)

def test_lattice():
    u = focus_field_lattice((128,) * 3, (.1,) * 3)



