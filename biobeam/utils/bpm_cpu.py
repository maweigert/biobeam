"""


mweigert@mpi-cbg.de

"""

from __future__ import absolute_import
from __future__ import print_function
import bpm
from bpm.bpmclass.bpm3d import Bpm3d
from numpy import *
import numpy as np
import pylab
import gputools
from time import time
from six.moves import range

if __name__ == '__main__':

    Nx, Nz = 256,512
    dx = .5

    #n = 0.05*dx*np.random.uniform(0,1.,(Nz,Nx,Nx))

    #n =  .05*gputools.perlin3((Nx,Nx,Nz),(4.,)*3)
    n = np.zeros((Nz,Nx,Nx))
    m = Bpm3d((Nx,Nx,Nz),(dx,)*3)

    u = ones((Nz,Nx,Nx),complex64)

    u[0] = bpm.psf_u0((Nx,Nx),(dx,dx),dx*Nz/2.,NA=.1)
    u[0] *= 1./np.sqrt(np.mean(abs(u[0])**2))
    t = time()

    x = np.linspace(-Nx,Nx,Nx+1)[:Nx]
    Y,X = np.meshgrid(x,x,indexing = "ij")
    h0 = np.exp(-.1*(X**2+Y**2))
    h0 *= 1./sum(h0)
    h0_f = np.fft.fftn(h0)

    for i in range(Nz-1):
        _u = u[i]
        _u = fft.fftn(_u)
        _u *= m._H
        _u = fft.ifftn(_u)
        #_u *= exp(-dx*1.j*m.k0*n[i+1])

        en0 = np.abs(_u)**2
        en = np.real(np.fft.fftshift(np.fft.fftn(np.fft.ifftn(en0)*h0_f)))
        _u *= np.sqrt((en+1.e-8)/(en0+1.e-8))

        # if i>0:
        #     break
        u[i+1] = _u

    print("time: %.1f ms"%(1000*(time()-t)))

    pylab.clf()
    pylab.plot(mean(abs(u)**2,(1,2)))
    pylab.show()