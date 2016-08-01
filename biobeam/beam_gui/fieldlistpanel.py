"""


mweigert@mpi-cbg.de

"""

from PyQt4 import QtGui, QtCore
from biobeam.beam_gui.fieldpanel import FieldPanel
from biobeam.beam_gui.fieldstate import CylindricalState, BeamState, LatticeState


class FieldListPanel(QtGui.QWidget):
    _stateChanged = QtCore.pyqtSignal(bool)

    def __init__(self):
        super(FieldListPanel, self).__init__()

        self.resize(50, 300)
        self.initUI()

    def initUI(self):

        vbox = QtGui.QVBoxLayout()

        self.fields = [CylindricalState(),
                       BeamState(),
                       LatticeState()]

        self.combo = QtGui.QComboBox()

        for field in self.fields:
            self.combo.addItem(field.name)

        vbox.addWidget(self.combo)

        self.panels = []
        for field in self.fields:
            pan = FieldPanel(field)
            pan._stateChanged.connect(self._stateChanged.emit)
            pan.hide()
            self.panels.append(pan)
            vbox.addWidget(pan)

        self.panels[0].show()

        vbox.addStretch()
        self.setLayout(vbox)

        self.combo.currentIndexChanged.connect(self.onIndexChanged)

        self.setStyleSheet("""
        QFrame,QLabel,QWidget, QComboBox, QLineEdit, QTextEdit {
        color: white;
        background-color:black;
        }
        """)

    def onIndexChanged(self, index):
        for pan in self.panels:
            pan.hide()
        self.panels[index].show()
        self._stateChanged.emit(True)


if __name__=='__main__':
    import sys

    app = QtGui.QApplication(sys.argv)

    win = FieldListPanel()

    win.show()

    win.raise_()

    sys.exit(app.exec_())
