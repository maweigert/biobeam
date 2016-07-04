"""
the main class for gpu accelerated bpm propagation

mweigert@mpi-cbg.de

"""

import numpy as np
from gputools import OCLArray, OCLImage, OCLProgram, get_device
from gputools import fft, fft_plan
from gputools import OCLReductionKernel
from bpm.utils import StopWatch, absPath

from bpm3d import _Bpm3d_Base


def absPath(myPath):
    import sys
    import os
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        return os.path.join(base_path, os.path.basename(myPath))
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, myPath)


class _Bpm3d_OCL(_Bpm3d_Base):
    """ OpenCL implementation
    """


    def _setup_impl(self):
        """setting up the gpu buffers and kernels
        """




        # self.reduce_kernel = OCLReductionKernel(
        # np.float32, neutral="0",
        #     reduce_expr="a+b",
        #     map_expr="weights[i]*cfloat_abs(field[i]-(i==0)*plain)*cfloat_abs(field[i]-(i==0)*plain)",
        #     arguments="__global cfloat_t *field, __global float * weights,cfloat_t plain")

    def _propagate_single(self, u0 = None,
                          return_full = True,
                          return_intensity = False,
                          absorbing_width = 0, **kwargs):
        """
        :param u0: initial complex field distribution, if None, plane wave is assumed
        :param kwargs:
        :return:
        """


        #plane wave if none
        if u0 is None:
            u0 = np.ones(self.size2d[::-1],np.complex64)


        Nx,Ny,Nz = self.size
        dx, dy, dz = self.units

        plane_g = OCLArray.from_array(u0.astype(np.complex64,copy = False))


        if return_full:
            if return_intensity:
                u_g = OCLArray.empty((Nz,Ny,Nx),dtype=np.float32)
                self.bpm_program.run_kernel("fill_with_energy",(Nx*Ny,),None,
                                   u_g.data,plane_g.data,np.int32(0))

            else:
                u_g = OCLArray.empty((Nz,Ny,Nx),dtype=np.complex64)
                u_g[0] = plane_g



        for i in range(Nz-1):
            fft(plane_g,inplace = True, plan  = self._plan)

            self.bpm_program.run_kernel("mult",(Nx*Ny,),None,
                               plane_g.data,self._H_g.data)

            fft(plane_g,inplace = True, inverse = True,  plan  = self._plan)

            if self.dn is not None:
                if self._is_complex_dn:
                    kernel_str = "mult_dn_complex"
                else:
                    kernel_str = "mult_dn"


                self.bpm_program.run_kernel(kernel_str,(Nx,Ny,),None,
                                   plane_g.data,self.dn_g.data,
                                   np.float32(self.k0*dz),
                                   np.int32(Nx*Ny*(i+1)),
                               np.int32(absorbing_width))
            if return_full:
                if return_intensity:
                    self.bpm_program.run_kernel("fill_with_energy",(Nx*Ny,),None,
                                   u_g.data,plane_g.data,np.int32((i+1)*Nx*Ny))

                else:
                    u_g[i+1] = plane_g

        if return_full:
            u = u_g.get()
        else:
            u = plane_g.get()


        return u


    def __repr__(self):
        return "Bpm3d class with size %s and units %s"%(self.size,self.units)


if __name__ == '__main__':

    from time import time

    size = (256,256,500)
    size = (512,)*3
    size = (512,512,400)

    dn = np.zeros(size[::-1],np.float32)
    ss = [slice(2*s/5,3*s/5) for s in size[::-1]]
    dn[ss] = 0.1
    t = time()
    m = _Bpm3d_OCL(size,(.1,)*3, dn = dn)
    print "%.2f ms"%(1000*(time()-t))

    t = time()
    u = m._propagate_single(absorbing_width=0,
                            return_full=False,
                            return_intensity=True)
    print "%.2f ms"%(1000*(time()-t))
