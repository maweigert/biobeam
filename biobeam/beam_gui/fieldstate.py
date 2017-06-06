"""


mweigert@mpi-cbg.de

"""

from __future__ import absolute_import
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets



class FieldState(QtCore.QObject):
    def __init__(self, name="name", description="description", kwargs={}):
        super(FieldState, self).__init__()
        self.name = name
        self.description = description
        self.kwargs = kwargs

    def _get_input_field(self, m):
        pass


class CylindricalState(FieldState):
    def __init__(self, NA=.2):
        super(CylindricalState, self).__init__(name="cylindrical",
                                               description="cylindrical lens lightsheet",
                                               kwargs={"NA": NA,
                                                       "y": 0,
                                                       "z": 0.}
                                               )

    def _get_input_field(self, m):
        NA = self.kwargs["NA"]
        y = self.kwargs["y"]
        z = self.kwargs["z"]+.5*m.size[-1]
        return m.u0_cylindrical(center=(0, y), zfoc=z, NA=NA)


class BeamState(FieldState):
    def __init__(self, NA=.2):
        super(BeamState, self).__init__(name="gaussian/bessel beam",
                                        description="gaussian bessel beam ",
                                        kwargs={"NA": NA,
                                                "y": 0,
                                                "z": 0.}
                                        )

    def _get_input_field(self, m):
        NA = self.kwargs["NA"]
        y = self.kwargs["y"]
        z = self.kwargs["z"]+.5*m.size[-1]

        return m.u0_beam(center=(0, y), zfoc=z, NA=NA)


class LatticeState(FieldState):
    def __init__(self, NA1=.3, NA2=.4, kpoints=6, sigma=0.1):
        super(LatticeState, self).__init__(name="bessel lattice",
                                           description="bessel lattice beam ",
                                           kwargs={"NA1": NA1,
                                                   "NA2": NA2,
                                                   "sigma": sigma,
                                                   "kpoints": kpoints,
                                                   "y": 0,
                                                   "z": 0.}
                                           )

    def _get_input_field(self, m):
        NA1 = self.kwargs["NA1"]
        NA2 = self.kwargs["NA2"]
        kpoints = self.kwargs["kpoints"]
        sigma = self.kwargs["sigma"]
        y = self.kwargs["y"]
        z = self.kwargs["z"]+.5*m.size[-1]

        return m.u0_lattice(center=(0, y), zfoc=z, NA1=NA1, NA2=NA2, sigma=sigma, kpoints=kpoints)
