"""


mweigert@mpi-cbg.de

"""

#!/usr/bin/env python



import os
import numpy as np
import sys

from PyQt4 import QtCore
from PyQt4 import QtGui

from spimagine.gui.mainwidget import MainWidget
from spimagine import NumpyData, DataModel
from biobeam.beam_gui.bpm3d_img import Bpm3d_img

import logging
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)


from prop_panel import PropPanel

class BeamGui(QtGui.QWidget):

    def __init__(self, dn, size, parent = None):
        super(BeamGui,self).__init__(parent)

        self.myparent = parent

        self.canvas = MainWidget(self)


        self.prop_panel = PropPanel()
        #self.prop_panel.prop_button.clicked.connect(self.propagate)
        #self.prop_panel._yposChanged.connect(self.on_ypos)
        self.prop_panel._propChanged.connect(self.on_prop_changed)


        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.canvas, stretch=3)
        hbox.addWidget(self.prop_panel)

        self.setLayout(hbox)

        self.resize(800,800)

        self.setStyleSheet("""
        background-color:black;
        color:black;
        """)

        self.properties = {"NA" : .2,
                         "beam_type" : "cylindrical",
                           "ypos":0}

        self.prop_panel.edit.setText(str(self.properties))

        self.reset_dn(dn, size)


    def reset_dn(self, dn , size, simul_z = 1, simul_xy = None):


        if simul_xy is None:
            simul_xy = dn.shape[1:][::-1]

        simul_z = 2
        simul_xy = (512,)*2

        self.bpm = Bpm3d_img(size = size, dn = dn,
                             lam = .5,
                             simul_z=simul_z,simul_xy=simul_xy)

        z = np.zeros_like(dn)
        z[0,0,0] = 1.

        self.canvas.setModel(DataModel(NumpyData(z)))



    def on_prop_changed(self,s):
        try:
            d = eval(str(s))
        except:
            print "could not parse"
            print s
            self.prop_panel.edit.setText(str(self.properties))
            return

        self.properties.update(d)

        self.propagate()


    def propagate(self):
        im = self.canvas.glWidget.renderer.dataImg

        NA = self.properties["NA"]

        if self.properties["beam_type"] == "beam":
            u0  = self.bpm.u0_beam(NA= NA)
        elif self.properties["beam_type"] == "cylindrical":
            u0  = self.bpm.u0_cylindrical(NA= NA)
        else:
            u0  = None

        if not u0 is None:
            u0 = np.roll(u0,self.properties["ypos"],0)

        self.bpm._propagate_to_img(u0 =  u0,im = im)

        self.canvas.glWidget.refresh()







def test_dn():

    x = np.linspace(-1,1,256)
    Z,Y,X = np.meshgrid(x,x,x)
    R = np.sqrt(Z**2+Y**2+X**2)

    dn = .1*(R<.4)
    return dn



if __name__ == '__main__':

    if not "dn" in locals():
        dn = test_dn()


    app = QtGui.QApplication(sys.argv)

    win = BeamGui(dn = dn, size = (50,)*3)


    win.show()

    win.raise_()


    win.propagate()


    sys.exit(app.exec_())

