from volust.volgpu import *
from volust.volgpu.oclalgos import *

import itertools
import numpy as np

import time


import os
import sys

from numpy import *
from scipy.special import *

def absPath(myPath):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        return os.path.join(base_path, os.path.basename(myPath))
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, myPath)

def test_bessel(n,x):
    x_g = OCLArray.from_array(x.astype(float32))
    res_g = OCLArray.empty_like(x.astype(float32))
    
    p = OCLProgram(absPath("bessel.cl"))
    p.run_kernel("bessel_fill",x_g.shape,None,
                 x_g.data,res_g.data,int32(n))

    return res_g.get()


if __name__ == "__main__":
   

    x = np.linspace(0,20,200)

    y = test_bessel(2,x)
