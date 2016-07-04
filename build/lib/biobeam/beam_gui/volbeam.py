"""


mweigert@mpi-cbg.de

"""


import sys
import numpy as np
import os

from PyQt4 import QtCore, QtGui

from collections import OrderedDict


from biobeam.beam_gui.beam_gui import BeamGui

_MAIN_APP = None

def getCurrentApp():
    app = QtGui.QApplication.instance()

    if not app:
        app = QtGui.QApplication(sys.argv)

    global _MAIN_APP
    _MAIN_APP = app
    return _MAIN_APP


def volbeam(dn = None,size = None,
            simul_xy  = None,
            simul_z = None,
            blocking = False,
            raise_window = True):


    if dn is None and size is None:
        dn = np.zeros((256,)*3,np.float32)
        dn[0,0,0] = 0.0000001
        size = (200,)*3
        simul_xy = (512,)*2
        simul_z = 2


    app = getCurrentApp()


    window = BeamGui(dn = dn.astype(np.float32),
                     size = size,
                     simul_xy = simul_xy,
                     simul_z=simul_z)

    window.show()

    window.propagate()
    if raise_window:
        window.raise_()

    if blocking:
        getCurrentApp().exec_()
    else:
        return window


def qt_exec():
    getCurrentApp().exec_()



if __name__ == '__main__':


    dn = np.random.uniform(0,.1,(128,)*3)

    volbeam(dn,size =(40,)*3, simul_xy=(256,)*2,simul_z = 2)
