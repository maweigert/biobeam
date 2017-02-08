"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division

from biobeam import focus_field_beam

# e.g. psf of a bessel beam with
# annulus 0.4<rho<0.45 in a volume
field = focus_field_beam(shape = (256,256,256),\
	units = (0.1,0.1, 0.1),NA = 0.4)