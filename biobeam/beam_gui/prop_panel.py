"""
A widget displaying propties, e.g. for the beam input classes

mweigert@mpi-cbg.de

"""

from __future__ import absolute_import
from PyQt4 import QtGui, QtCore
from spimagine.gui.gui_utils import createStandardCheckbox


class MyEdit(QtGui.QTextEdit):
    returnPressed = QtCore.pyqtSignal()

    def __init__(self, parent):
        super(MyEdit, self).__init__(parent)

    def keyPressEvent(self, e):
        if e.key()==QtCore.Qt.Key_Return:
            self.returnPressed.emit()
        else:
            super(MyEdit, self).keyPressEvent(e)


class PropPanel(QtGui.QWidget):
    _yposChanged = QtCore.pyqtSignal(int)
    _propChanged = QtCore.pyqtSignal(str)

    def __init__(self):
        super(PropPanel, self).__init__()

        self.resize(50, 300)
        self.initUI()

    def initUI(self):

        self.check_dn = createStandardCheckbox(self)

        self.disp_dn = createStandardCheckbox(self)

        gridBox = QtGui.QGridLayout()
        gridBox.addWidget(self.check_dn, 0, 0)
        gridBox.addWidget(QtGui.QLabel("use dn"), 0, 1)


        gridBox.addWidget(self.disp_dn, 1, 0)
        gridBox.addWidget(QtGui.QLabel("show dn"), 1, 1)



        self.setLayout(gridBox)

        self.setStyleSheet("""
        QFrame,QLabel,QLineEdit, QTextEdit {
        color: white;
        }
        """)


if __name__=='__main__':
    import sys

    app = QtGui.QApplication(sys.argv)

    win = PropPanel()

    win.show()

    win.raise_()

    sys.exit(app.exec_())
