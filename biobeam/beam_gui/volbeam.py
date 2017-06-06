"""


mweigert@mpi-cbg.de

"""

from __future__ import absolute_import
import sys
import numpy as np
import os
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets

from collections import OrderedDict
from biobeam.beam_gui.beam_gui import BeamGui

_MAIN_APP = None


def getCurrentApp():
    app = QtWidgets.QApplication.instance()

    if not app:
        app = QtWidgets.QApplication(sys.argv)

    global _MAIN_APP
    _MAIN_APP = app
    return _MAIN_APP


def volbeam(dn=None, size=None,
            simul_xy=None,
            simul_z=None,
            blocking=False,
            raise_window=True):
    if dn is None and size is None:
        dn = np.zeros((256,)*3, np.float32)
        dn[0, 0, 0] = 0.0000001
        size = (200,)*3
        simul_xy = (512,)*2
        simul_z = 2

    app = getCurrentApp()

    window = BeamGui(dn=dn.astype(np.float32),
                     size=size,
                     simul_xy=simul_xy,
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


def get_volume(N=256):
    from gputools import perlin3

    print("creating volume...")

    _x = np.linspace(-1,1,N)
    R = np.sqrt(np.sum([_X**2 for _X in np.meshgrid(_x,_x,_x,indexing = "ij")],axis =0))

    dn = .1*(R<.2)
    dn += .1*perlin3((N,)*3, scale = 4)

    print("... done")

    return dn


if __name__=='__main__':
    dn = get_volume(256)

    volbeam(dn, size=(40,)*3, blocking = True)
