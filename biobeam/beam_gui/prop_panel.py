"""


mweigert@mpi-cbg.de

"""

from PyQt4 import  QtGui, QtCore


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
        #self.prop_button = QtGui.QPushButton("propagate",self)

        #self.prop_button.setStyleSheet("background-color: black;color:white;")


        # self.edit_ypos = QtGui.QLineEdit("1")
        # self.edit_ypos .setValidator(QtGui.QIntValidator())
        # self.edit_ypos.returnPressed.connect(lambda :self._yposChanged.emit(int(self.edit_ypos.text())))


        self.edit = MyEdit("1")

        self.edit.returnPressed.connect(lambda :self._propChanged.emit(self.edit.toPlainText()))

        vbox.addWidget(lab)
        #vbox.addWidget(self.edit_ypos)
        vbox.addWidget(self.edit)
        #vbox.addWidget(self.prop_button)
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

