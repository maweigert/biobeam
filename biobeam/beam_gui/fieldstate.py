"""


mweigert@mpi-cbg.de

"""

from PyQt4 import QtCore


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
                                                       "z": 0}
                                               )

    def _get_input_field(self, m):
        return m.u0_cylindrical(center=(0, 0), zfoc=None, NA=self.kwargs["NA"])


class BeamState(FieldState):
    def __init__(self, NA=.2):
        super(BeamState, self).__init__(name="gaussian/bessel beam",
                                        description="gaussian bessel beam ",
                                        kwargs={"NA": NA,
                                                "y": 0,
                                                "z": 0}
                                        )

    def _get_input_field(self, m):
        return m.u0_beam(center=(0, 0), zfoc=None, NA=self.kwargs["NA"])


class LatticeState(FieldState):
    def __init__(self, NA=.2):
        super(LatticeState, self).__init__(name="bessel lattice",
                                           description="bessel lattice beam ",
                                           kwargs={"NA": NA,
                                                   "y": 0,
                                                   "z": 0}
                                           )

    def _get_input_field(self, m):
        return m.u0_lattice(center=(0, 0), zfoc=None, NA=self.kwargs["NA"])
