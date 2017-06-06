"""
The Beam GUI class that provides a simple Spimagine based Viewer
to allow for some simple and easy light propagation demos


mweigert@mpi-cbg.de

"""

# !/usr/bin/env python



from __future__ import absolute_import
from __future__ import print_function
import os
import numpy as np
import sys
from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from spimagine.gui.mainwidget import MainWidget
from spimagine import NumpyData, DataModel, read3dTiff
from biobeam.beam_gui.bpm3d_img import Bpm3d_img
from biobeam.beam_gui.fieldlistpanel import FieldListPanel
from biobeam.beam_gui.prop_panel import PropPanel
import logging
from six.moves import zip

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class BeamGui(QtWidgets.QWidget):
    def __init__(self, dn, size, simul_xy=None, simul_z=None, parent=None):
        super(BeamGui, self).__init__(parent)

        self.myparent = parent

        self.canvas = MainWidget(self)

        self.isFullScreen = False
        self.prop_panel = PropPanel()
        self.prop_panel.disp_dn.stateChanged.connect(self.view_dn)
        self.prop_panel.check_dn.stateChanged.connect(self.check_dn)

        self.prop_panel._propChanged.connect(self.on_prop_changed)

        self.field_list_panel = FieldListPanel()
        self.field_list_panel._stateChanged.connect(self.propagate)

        self.free_prop = True

        vbox = QtWidgets.QVBoxLayout()

        vbox.addWidget(self.prop_panel)
        vbox.addWidget(self.field_list_panel)

        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.canvas, stretch=3)
        hbox.addLayout(vbox)

        self.setLayout(hbox)

        self.resize(800, 800)

        self.setStyleSheet("""
        background-color:black;
        color:black;
        """)

        self.properties = {"NA": .2,
                           "beam_type": "cylindrical",
                           "ypos": 0}

        #self.prop_panel.edit.setText(str(self.properties))

        self.reset_dn(dn, size, simul_xy=simul_xy, simul_z=simul_z)

        self.field_list_panel._stateChanged.connect(self.propagate)

    def check_dn(self):
        if self.prop_panel.check_dn.checkState():
            self.free_prop = False
        else:
            self.free_prop = True
        self.propagate()

    def view_dn(self):
        if self.prop_panel.disp_dn.checkState():
            self.canvas.glWidget.renderer.dataImg = self.bpm._im_dn
            self.canvas.glWidget.refresh()

        else:
            self.canvas.glWidget.renderer.dataImg = self.bpm.result_im

            self.canvas.glWidget.refresh()

    def reset_dn(self, dn, size, simul_z=None, simul_xy=None):

        if simul_z is None:
            simul_z = 1
        if simul_xy is None:
            simul_xy = dn.shape[1:][::-1]

        # simul_z = 2
        # simul_xy = (1024,)*2
        # simul_z = 2
        # simul_xy = (512,)*2

        self.bpm = Bpm3d_img(size=size, dn=dn,
                             lam=.5,
                             simul_z=simul_z, simul_xy=simul_xy)

        if not dn is None:
            self.dn_max = np.amax(dn)

        z = np.zeros_like(dn)
        z[0, 0, 0] = 1.

        units = [s/(n-1.) for s, n in zip(size, dn.shape[::-1])]
        self.canvas.setModel(DataModel(NumpyData(z, stackUnits=units)))

    def on_prop_changed(self, s):
        try:
            d = eval(str(s))
        except:
            print("could not parse")
            print(s)
            self.prop_panel.edit.setText(str(self.properties))
            return

        self.properties.update(d)

        self.propagate()

    # def propagate(self):
    #     im = self.canvas.glWidget.renderer.dataImg
    #
    #     NA = self.properties["NA"]
    #
    #     if self.properties["beam_type"] == "beam":
    #         u0  = self.bpm.u0_beam(NA= NA)
    #     elif self.properties["beam_type"] == "cylindrical":
    #         u0  = self.bpm.u0_cylindrical(NA= NA)
    #     elif self.properties["beam_type"] == "lattice":
    #         try:
    #             NA1, NA2 = NA
    #             print NA1, NA2
    #         except:
    #             QtGui.QMessageBox.information(None, 'Error',
    #         "NA should be a list of two NAs, e.g. (.4,.5)")
    #
    #         sigma = self.properties.get("sigma",.1)
    #         u0  = self.bpm.u0_lattice(NA1= NA1, NA2 = NA2, sigma = sigma)
    #
    #     elif self.properties["beam_type"] == "plane":
    #         u0  = None
    #
    #     else:
    #         u0  = None
    #
    #     if not u0 is None:
    #         u0 *= 0.01/np.sqrt(np.mean(np.abs(u0)**2))
    #         # u0 *= np.sqrt(self.dn_max)/np.sqrt(np.mean(np.abs(u0)**2))
    #         u0 = np.roll(u0,self.properties["ypos"],0)
    #
    #     print np.amax(np.abs(np.imag(self.bpm.dn)))
    #
    #     print self.free_prop
    #     self.bpm._propagate_to_img(u0 =  u0,im = im, free_prop = self.free_prop)
    #
    #     self.canvas.glWidget.refresh()


    def propagate(self):
        print("propagate!")
        im = self.canvas.glWidget.renderer.dataImg

        field = self.field_list_panel.fields[self.field_list_panel.combo.currentIndex()]

        u0 = field._get_input_field(self.bpm)

        if not u0 is None:
            u0 *= 0.01/np.sqrt(np.mean(np.abs(u0)**2))

        self.bpm._propagate_to_img(u0=u0, im=im, free_prop=self.free_prop)

        self.canvas.glWidget.refresh()

    def mouseDoubleClickEvent(self, event):
        if self.isFullScreen:
            self.showNormal()
        else:
            self.showFullScreen()
        self.isFullScreen = not self.isFullScreen

        # self.canvas.mouseDoubleClickEvent(event)



def sphere_dn(N = 128):
    import gputools
    print("creating sphere with N = ", N)
    x = np.linspace(-1, 1, N)
    Z, Y, X = np.meshgrid(x, x, x)
    R = np.sqrt(Z**2+Y**2+X**2)

    noise = gputools.perlin3((N,)*3,scale= 10)
    dn = .1*noise*(R<.4)*(1-7.j)


    return dn


if __name__=='__main__':
    import sys

    if len(sys.argv)>1:
        if sys.argv[1].isdigit():
            dn = sphere_dn(int(sys.argv[1]))
        else:
            dn = read3dTiff(sys.argv[1])
    else:
        dn = sphere_dn(128)

    app = QtWidgets.QApplication(sys.argv)

    win = BeamGui(dn=dn, size=(100,)*3)
    # win = BeamGui(dn = dn, size = (50,50,25))


    win.show()

    win.raise_()

    win.propagate()

    #sys.exit(app.exec_())
    #sys.exit(app.exec_())
