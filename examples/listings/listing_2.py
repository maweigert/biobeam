"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division


from biobeam import Bpm3d
from gputools import perlin3

# set up the refractive index distribution
dn  = 0.03*perlin3((256,256,256),scale = 4.)


# set up the propagator class
m = Bpm3d(dn = dn, size = (70,70,70),
          lam = .5,n0 = 1.33)

# propagate the light field...
field = m.propagate(u0 = m.u0_beam(NA = (0.4,.41)))
