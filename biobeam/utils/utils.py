
from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import time
import numpy  as np
import six
from six.moves import range
                
class StopWatch(object):
    """ stops time in miliseconds

    s = StopWatch()

    s.tic()

    foo()
    
    print t.toc()
    """
    def __init__(self):
        self.times  = dict()
        self._dts  = dict()

    def tic(self,key = ""):
        self._dts[key] = time.time()

    def toc(self,key = ""):
        self.times[key] = 1000.*(time.time()- self._dts[key])
        return self.times[key]

    def __getitem__(self, key,*args):
        return self.times.__getitem__(key,*args)

    def __setitem__(self, key,val):
        self.times[key] = val


    def __repr__(self):
        return "\n".join(["%s:\t%.3f ms"%(str(k),v) for k,v in six.iteritems(self.times)])

def absPath(myPath):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        return os.path.join(base_path, os.path.basename(myPath))
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, myPath)

def pad_to_shape(h,dshape, mode = "constant"):
    if h.shape == dshape:
        return h

    diff = np.array(dshape)- np.array(h.shape)
    #first shrink
    slices  = [slice(-x//2,x//2) if x<0 else slice(None,None) for x in diff]
    res = h[slices]
    #then padd
    return np.pad(res,[(d//2,d-d//2) if d>0 else (0,0) for d in diff],mode=mode)

def _is_power2(n):
    return _next_power_of_2(n) == n

def _next_power_of_2(n):
    return int(2**np.ceil(np.log2(n)))

def pad_to_power2(data, axis = None,mode="constant"):
    """pads shape to power of two
    if axis is None all axis are padded, otherwise just
    the ones given e.g axis=[0,2]
    """
    if axis is None:
        axis = list(range(data.ndim))
    if np.all([_is_power2(n) for n in np.array(data.shape)[axis]]):
        return data
    else:
        newShape = [_next_power_of_2(n) if i in axis else n for i,n in enumerate(data.shape)]
        return pad_to_shape(data,newShape,mode)

if __name__ == '__main__':



    print("absPath: ", absPath(".")) 

    d = np.ones((1,7,12))

    d2 = pad_to_power2(d,axis=[0,1])

    print(d2.shape)
