import sys

import numpy as np

from PyQt4 import QtCore, QtGui, Qt


def absPath(myPath):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    import os
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        return os.path.join(base_path, os.path.basename(myPath))
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, myPath)


checkBoxStyleStr = """
    QCheckBox::indicator:checked {
    background:black;
    border-image: url(%s);}
    QCheckBox::indicator:unchecked {
    background:black;
    border-image: url(%s);}
    """

def createImgCheckBox(fName_active,fName_inactive):
    checkBox = QtGui.QCheckBox()
    checkBox.setStyleSheet(
            checkBoxStyleStr%(absPath(fName_active),absPath(fName_inactive)))
    return checkBox



class ParameterView(QtGui.QWidget):
    """
        state.name defines the name
        state.kwargs defines the parameters
    """
    _stateChanged = QtCore.pyqtSignal(bool)

    def __init__(self, state = None):
        """ state.kwargs defines the parameters
        """

        super(QtGui.QWidget,self).__init__()

        self.resize(300, 300)
        self.set_state(state)
        self.initUI()


    def set_state(self, state):
        self._state = state


    def initUI(self):
        if self._state is None:
            return

        vbox = QtGui.QVBoxLayout()

        gridBox = QtGui.QGridLayout()

        gridBox.addWidget(QtGui.QLabel(self._state.name),0,0)

        for i, (key, val)  in enumerate(self._state.kwargs.iteritems()):
            dtype = type(val)
            if dtype == bool:
                check = QtGui.QCheckBox("",self)
                check.stateChanged.connect(self.set_state_attr_check(check,key,val))
                gridBox.addWidget(QtGui.QLabel(key),i+1,0)
                gridBox.addWidget(check,i+1,1)

            elif dtype in (int,float,np.int,np.float):
                edit = QtGui.QLineEdit(str(val))
                edit.setValidator(QtGui.QDoubleValidator())
                edit.returnPressed.connect(self.set_state_attr_edit_single(edit,key,dtype))
                gridBox.addWidget(QtGui.QLabel(key),i+1,0)
                gridBox.addWidget(edit,i+1,1)
            elif dtype in (list,tuple):
                edit = QtGui.QLineEdit(",".join([str(v) for v in val]))
                edit.returnPressed.connect(self.set_state_attr_edit_list(edit,key,dtype))
                gridBox.addWidget(QtGui.QLabel(key),i+1,0)
                gridBox.addWidget(edit,i+1,1)

        vbox.addLayout(gridBox)
        vbox.addStretch()

        self.setLayout(vbox)
        self.setSizePolicy(Qt.QSizePolicy.Minimum,Qt.QSizePolicy.Minimum)
        self.setStyleSheet("""
        QFrame,QWidget,QLabel,QLineEdit {
        color: white;
        background-color:black;
        }

        """)





    def set_state_attr_edit_single(self,obj, key , dtype):
        def func():
            print "setting", key , obj.text()
            setattr(self._state,key, dtype(obj.text()))
            self._stateChanged.emit(-1)
        return func

    def set_state_attr_edit_list(self,obj, key , dtype):
        def func():
            setattr(self._state,key, [float(_s.strip()) for _s in str(obj.text()).strip().split(",") if len(_s.strip())>0])
            self._stateChanged.emit(-1)
        return func

    def set_state_attr_check(self,obj, key , dtype):
        def func():
            print "setting", key , obj.checkState() !=0
            setattr(self._state,key, obj.checkState() !=0)
            self._stateChanged.emit(-1)
        return func



class State(object):
    def __init__(self, name, kwargs= {}):
        self.name = name
        self.kwargs = kwargs



if __name__ == '__main__':


    app = QtGui.QApplication(sys.argv)



    win = ParameterView(State("hallo",{"x":0., "y":[0,1,2]}))
    win.show()
    win.raise_()

    sys.exit(app.exec_())

