"""


mweigert@mpi-cbg.de

"""

from PyQt4 import  QtGui, QtCore
from spimagine.gui.gui_utils import createStandardCheckbox


class MyEdit(QtGui.QTextEdit):
    returnPressed = QtCore.pyqtSignal()
    def __init__(self, parent):
        super (MyEdit, self).__init__(parent)



    def keyPressEvent(self, e):
        if e.key()==QtCore.Qt.Key_Return:
            self.returnPressed.emit()
        else:
            super (MyEdit, self).keyPressEvent(e)


class PropPanel(QtGui.QWidget):
    _yposChanged = QtCore.pyqtSignal(int)
    _propChanged = QtCore.pyqtSignal(str)
    def __init__(self):
        super(PropPanel,self).__init__()

        self.resize(50, 300)
        self.initUI()


    def initUI(self):


        vbox = QtGui.QVBoxLayout()


        lab = QtGui.QLabel("properties")

        self.check_dn = createStandardCheckbox(self)

        self.disp_dn = createStandardCheckbox(self)

        self.edit = MyEdit("1")

        self.edit.returnPressed.connect(lambda :self._propChanged.emit(self.edit.toPlainText()))

        vbox.addWidget(lab)
        vbox.addWidget(self.check_dn)
        vbox.addWidget(self.disp_dn)

        vbox.addWidget(self.edit)
        vbox.addStretch()
        self.setLayout(vbox)

        self.setStyleSheet("""
        QFrame,QLabel,QLineEdit, QTextEdit {
        color: white;
        }
        """)


if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)

    win = PropPanel()


    win.show()

    win.raise_()


    sys.exit(app.exec_())

