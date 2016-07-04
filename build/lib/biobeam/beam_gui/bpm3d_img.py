"""

a special class that writes its output directly to a texture

mweigert@mpi-cbg.de

"""

import numpy as np
from gputools import OCLArray, OCLImage, OCLProgram, get_device
from gputools import fft, fft_plan


#from biobeam.psf.psf_functions import psf_u0, psf_cylindrical_u0
#from gputools import OCLReductionKernel


from biobeam import Bpm3d




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


class Bpm3d_img(Bpm3d):
    """

    """

    _real_type = np.float32
    _complex_type = np.complex64

    def __init__(self, *args, **kwargs):
        kwargs["enforce_subsampled"] = True
        super(Bpm3d_img,self).__init__(*args,**kwargs)
        self._is_subsampled = True

        self.result_im = OCLImage.empty(self.shape[::-1],dtype=np.float32)






    def _copy_down_img_to_img(self,im1, im2, zPos):
        Nx, Ny = self.shape[:2]
        self._kernel_im_to_im_intensity(self._queue, (Nx,Ny), None,
                                     im1, im2,
                                     np.int32(zPos))

    def _propagate_to_img(self, u0 = None, im = None, free_prop = False, **kwargs):
        """
        """


        free_prop = free_prop or (self.dn is None)

        res_type = Bpm3d._real_type


        if u0 is None:
            u0 = self.u0_plane()

        Nx, Ny, Nz = self.shape


        if im is None:
            im = self.result_im


        self._buf_plane.write_array(u0)

        #copy the first plane


        self._img_xy.copy_buffer(self._buf_plane)
        self._copy_down_img_to_img(self._img_xy,im,0)


        for i in xrange(Nz-1):
            for j in xrange(self.simul_z):

                fft(self._buf_plane, inplace = True, plan  = self._plan)
                self._mult_complex(self._buf_plane, self._buf_H)
                fft(self._buf_plane, inplace = True, inverse = True, plan  = self._plan)
                if not free_prop:
                    self._mult_dn(self._buf_plane,(i+(j+1.)/self.simul_z))

            self._img_xy.copy_buffer(self._buf_plane)
            self._copy_down_img_to_img(self._img_xy,im,i+1)

        return im

        def __repr__(self):
            return "Bpm3d class \nsize: \t%s\nshape:\t %s\nresolution: (%.4f,%.4f,%.4f)  "%(self.size,self.shape,self.dx,self.dy,self.dz)


if __name__ == '__main__':

    from time import time

    shape = (256,256,256)
    size = (40,40,40)
    #size = (200,200,200)

    if not "dn" in locals():
        xs = [np.linspace(-.5*s,.5*s,N) for s,N in zip(size,shape)]
        Xs = np.meshgrid(*xs[::-1],indexing="ij")
        R = np.sqrt(reduce(np.add,[_X**2 for _X in Xs]))
        dn = (.1-.001j)*(R<size[0]/4)
        #dn *= 0
        import gputools
        dn += .002*gputools.perlin3(shape, units = (1.9,)*3)
        #dn -= .001j*np.abs(gputools.perlin3(shape, units = (.3,.3,.3)))
        # dn = .07*gputools.perlin3(shape, units = (.2,.2,.2))

    m = Bpm3d_img(size = size,dn = dn,simul_z = 2, simul_xy=(512,512))


    t = time()
    u = m._propagate_to_img(free_prop=True)
