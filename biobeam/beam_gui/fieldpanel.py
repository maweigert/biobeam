from __future__ import absolute_import
from __future__ import print_function
import sys
import numpy as np
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5 import QtWidgets
import biobeam
import logging
import six

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


def createImgCheckBox(fName_active, fName_inactive):
    checkBox = QtGui.QCheckBox()
    checkBox.setStyleSheet(
        checkBoxStyleStr%(absPath(fName_active), absPath(fName_inactive)))
    return checkBox


class FieldPanel(QtWidgets.QWidget):
    """
        state.name defines the name
        state.kwargs defines the parameters
    """
    _stateChanged = QtCore.pyqtSignal(bool)

    def __init__(self, field):
        """ state_dict defines the parameters, e.g. state_dict = {"sigma":1.}
        """

        super(FieldPanel, self).__init__()

        self.resize(300, 300)
        self._name = field.name
        self._description = field.description
        self.set_state(field.kwargs)
        self.initUI()

    def set_state(self, state_dict):
        self._state = state_dict

    def initUI(self):
        if self._state is None:
            return

        vbox = QtWidgets.QVBoxLayout()

        gridBox = QtWidgets.QGridLayout()

        gridBox.addWidget(QtWidgets.QLabel(self._description), 0, 0)

        for i, (key, val) in enumerate(six.iteritems(self._state)):
            dtype = type(val)
            if dtype==bool:
                check = QtWidgets.QCheckBox("", self)
                check.stateChanged.connect(self.set_state_attr_check(check, key, val))
                gridBox.addWidget(QtWidgets.QLabel(key), i+1, 0)
                gridBox.addWidget(check, i+1, 1)

            elif dtype in (int, float, np.int, np.float):
                edit = QtWidgets.QLineEdit(str(val))
                edit.setValidator(QtGui.QDoubleValidator())
                edit.returnPressed.connect(self.set_state_attr_edit_single(edit, key, dtype))
                gridBox.addWidget(QtWidgets.QLabel(key), i+1, 0)
                gridBox.addWidget(edit, i+1, 1)
            elif dtype in (list, tuple):
                edit = QtWidgets.QLineEdit(",".join([str(v) for v in val]))
                edit.returnPressed.connect(self.set_state_attr_edit_list(edit, key, dtype))
                gridBox.addWidget(QtWidgets.QLabel(key), i+1, 0)
                gridBox.addWidget(edit, i+1, 1)

        vbox.addLayout(gridBox)
        vbox.addStretch()

        self.setLayout(vbox)
        self.setSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        self.setStyleSheet("""
        QFrame,QWidget,QLabel,QLineEdit {
        color: white;
        background-color:black;
        }

        """)

    def set_state_attr_edit_single(self, obj, key, dtype):
        def func():
            self._state[key] = dtype(obj.text())
            self._stateChanged.emit(-1)
            logger.debug(str(self._state))

        return func

    def set_state_attr_edit_list(self, obj, key, dtype):
        def func():
            self._state[key] = [float(_s.strip()) for _s in str(obj.text()).strip().split(",") if len(_s.strip())>0]
            self._stateChanged.emit(-1)
            logger.debug(str(self._state))

        return func

    def set_state_attr_check(self, obj, key, dtype):
        def func():
            self._state[key] = obj.checkState()!=0
            self._stateChanged.emit(-1)
            logger.debug(str(self._state))

        return func


if __name__=='__main__':
    from biobeam.beam_gui.fieldstate import CylindricalState, LatticeState
    app = QtWidgets.QApplication(sys.argv)

    c = LatticeState()

    win = FieldPanel(c)
    win.show()
    win.raise_()

    app.exec_()

    print(c.kwargs)
