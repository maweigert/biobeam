"""
the main class for gpu accelerated bpm propagation

mweigert@mpi-cbg.de

"""

import numpy as np
from gputools import OCLArray, OCLImage, OCLProgram, get_device
from gputools import fft, fft_plan


#from biobeam.psf.psf_functions import psf_u0, psf_cylindrical_u0
#from gputools import OCLReductionKernel


from bpm import psf_u0, psf_cylindrical_u0




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


class Bpm3d(object):
    """
    the main class for gpu accelerated bpm propagation
    """

    _real_type = np.float32
    _complex_type = np.complex64

    def __init__(self, size,
                 shape = None,
                 dn = None,
                 lam = .5,
                 n0 = 1.,
                 simul_xy = None,
                 simul_z =1,
                 n_volumes = 1,
                 enforce_subsampled = False,
                 fftplan_kwargs = {}):
        """

        :param size:  the size of the geometry in pixels (Nx,Ny,Nz)
        :param units: the physical units of each voxel in microns (dx,dy,dz)
        :param dn: refractive index distribution (can be given later)
        :param lam: the wavelength of light in microns
        :param n0:  the refractive index of the surrounding media
        :param n_volumes: splits the domain into chunks if GPU memory is not
                        large enough

        example:

        model = Bpm3d(size = (128,128,128),
                      units = (0.1,0.1,0.1),
                      lam = 0.5,
                      n0 = 1.33)
        """

        if shape is None and dn is None:
            raise ValueError("either shape or dn have to be given!")
        if not(shape is None or dn is None) and dn.shape != shape[::-1]:
            raise ValueError("shape != dn.shape")

        if shape is None:
            shape = dn.shape[::-1]

        self.n_volumes = n_volumes
        self.fftplan_kwargs = fftplan_kwargs

        if simul_xy is None:
            simul_xy = shape[:2]



        self._setup(shape = shape, size = size,  lam = lam, n0 = n0,
                    simul_xy = simul_xy,
                    simul_z = simul_z,
                    enforce_subsampled = enforce_subsampled )

        self._setup_dn(dn)



    def _copy_arr_with_correct_type(self,arr):
        """if arr is of the acceptable types, returns arr, else
        returns a copy with the acceptable type"""
        return arr.astype(Bpm3d._complex_type, copy = False) if np.iscomplexobj(arr) else arr.astype(Bpm3d._real_type, copy = False)

    def _setup_dn(self,dn):
        if dn is None:
            self.dn = None
        else:
            if dn.shape != self.shape[::-1]:
                raise ValueError("shape != dn.shape")
            self.dn = dn
            if self._is_subsampled:
                self._im_dn = OCLImage.from_array(self._copy_arr_with_correct_type(dn))
            else:
                self._buf_dn = OCLArray.from_array(self._copy_arr_with_correct_type(dn))



    def _setup(self, shape, size, lam, n0,
               simul_xy, simul_z, enforce_subsampled):
        """
            sets up the internal variables e.g. propagators etc...

            :param size:  the size of the geometry in pixels (Nx,Ny,Nz)
            :param units: the phyiscal units of each voxel in microns (dx,dy,dz)
            :param lam: the wavelength of light in microns
            :param n0:  the refractive index of the surrounding media,
            dn=None means free propagation
            :param use_fresnel_approx:  if True, uses fresnel approximation for propagator


        """


        self.shape = shape
        self.size = size
        self.lam = lam
        self.simul_xy = simul_xy
        self.simul_z = simul_z
        self.dx, self.dy = 1.*np.array(self.size[:2])/(np.array(self.simul_xy)-1)
        self.dz = 1.*self.size[-1]/(self.shape[-1]-1)/self.simul_z
        #self.dz = 1.*self.size[-1]/self.shape[-1]/self.simul_z
        self.n0 = n0
        self.k0 = 2.*np.pi/self.lam
        self._is_subsampled = enforce_subsampled or ((self.shape[:2] != self.simul_xy) or (simul_z>1))


        self._setup_gpu()


    def _setup_gpu(self):
        dev = get_device()
        self._queue = dev.queue
        self._ctx = dev.context
        prog = OCLProgram(absPath("kernels/bpm_3d_kernels.cl"))

        # the buffers/ images
        Nx, Ny = self.simul_xy
        Nx0, Ny0 = self.shape[:2]

        self._plan = fft_plan((Ny,Nx), **self.fftplan_kwargs)
        self._buf_plane = OCLArray.empty((Ny, Nx), np.complex64)
        self._buf_H = OCLArray.empty((Ny, Nx), np.complex64)

        self._img_xy = OCLImage.empty((Ny,Nx), dtype = np.float32,num_channels =2)

        # the kernels
        self._kernel_compute_propagator = prog.compute_propagator
        self._kernel_compute_propagator.set_scalar_arg_dtypes((None,)+(np.float32,)*5)

        self._kernel_mult_complex = prog.mult

        self._kernel_im_to_buf_field = prog.img_to_buf_field
        self._kernel_im_to_buf_intensity= prog.img_to_buf_intensity
        self._kernel_im_to_im_intensity= prog.img_to_img_intensity
        self._kernel_buf_to_buf_field = prog.buf_to_buf_field
        self._kernel_buf_to_buf_intensity = prog.buf_to_buf_intensity

        self._kernel_mult_dn_img_float = prog.mult_dn_image
        self._kernel_mult_dn_buf_float = prog.mult_dn
        self._kernel_mult_dn_img_complex = prog.mult_dn_image_complex
        self._kernel_mult_dn_buf_complex = prog.mult_dn_complex

        self._fill_propagator(self.n0)

    def _mult_dn(self, buf, zPos):
        if (self._is_subsampled and self._im_dn.dtype == Bpm3d._complex_type) or\
            (not self._is_subsampled and self._buf_dn.dtype == Bpm3d._complex_type):
            self._mult_dn_complex(buf,zPos)
        else:
            self._mult_dn_float(buf,zPos)


    def _mult_dn_float(self, buf, zPos):
        if self._is_subsampled:
            self._kernel_mult_dn_img_float(self._queue, self.simul_xy, None,
                                    buf.data,self._im_dn,
                                    np.float32(self.k0*self.dz),
                                    np.float32(zPos/(self.shape[-1]-1.)))
        else:
            Nx, Ny = self.shape[:2]
            self._kernel_mult_dn_buf_float(self._queue, self.shape[:2], None,
                                    buf.data,self._buf_dn.data,
                                    np.float32(self.k0*self.dz),
                                    np.int32(zPos*Nx*Ny))

    def _mult_dn_complex(self, buf, zPos):
        if self._is_subsampled:
            self._kernel_mult_dn_img_complex(self._queue, self.simul_xy, None,
                                    buf.data,self._im_dn,
                                    np.float32(self.k0*self.dz),
                                    np.float32(zPos/(self.shape[-1]-1.)))
        else:
            Nx, Ny = self.shape[:2]
            self._kernel_mult_dn_buf_complex(self._queue, self.shape[:2], None,
                                    buf.data,self._buf_dn.data,
                                    np.float32(self.k0*self.dz),
                                    np.int32(zPos*Nx*Ny))

    def _copy_down_img(self,im,buf,offset):
        Nx, Ny = self.shape[:2]
        if buf.dtype.type == Bpm3d._complex_type:
            self._kernel_im_to_buf_field(self._queue, (Nx,Ny), None,
                                     im, buf.data,
                                     np.int32(offset))
        elif buf.dtype.type == Bpm3d._real_type:
            self._kernel_im_to_buf_intensity(self._queue, (Nx,Ny), None,
                                     im, buf.data,
                                     np.int32(offset))
        else:
            assert False



    def _copy_down_buf(self,buf1,buf2,offset):
        Nx, Ny = self.shape[:2]

        if buf2.dtype.type == Bpm3d._complex_type:
            self._kernel_buf_to_buf_field(self._queue, (Nx*Ny,), None,
                                     buf1.data, buf2.data,
                                     np.int32(offset))
        elif buf2.dtype.type == Bpm3d._real_type:
            self._kernel_buf_to_buf_intensity(self._queue, (Nx*Ny,), None,
                                     buf1.data, buf2.data,
                                     np.int32(offset))
        else:
            assert False



    def _fill_propagator(self, n0):
        self._kernel_compute_propagator(self._queue, self.simul_xy, None,
                                        self._buf_H.data,
                                        n0, self.k0, self.dx, self.dy, self.dz)

    def _mult_complex(self, buf1, buf2):
        """buf1 *= buf2"""
        Nx, Ny = self.simul_xy
        self._kernel_mult_complex(self._queue,(Nx*Ny,), None,
                                        buf1.data,buf2.data)




    def u0_plane(self, phi = 0):
        return np.exp(1.j*phi)*np.ones(self.simul_xy[::-1],np.complex64)


    def u0_beam(self, zfoc = None, NA = .3):
        if zfoc is None:
            zfoc = .5*self.size[-1]
        return psf_u0(shape = self.simul_xy, units = (self.dx, self.dy),
                      zfoc = zfoc,NA = NA,
                      lam = self.lam, n0 = self.n0)

    def u0_cylindrical(self, zfoc =None ,  NA = .3):
        if zfoc is None:
            zfoc = .5*self.size[-1]

        return psf_cylindrical_u0(shape = self.simul_xy, units = (self.dx, self.dy),
                      zfoc = zfoc, NA = NA,
                      lam = self.lam, n0 = self.n0)



    def _propagate(self, u0 = None, offset = 0, **kwargs):
        """

        kwargs:
            return_result in ["field", "intens"]
            return_shape in ["last", "full"]
            free_prop = False|True
        """

        return_val = kwargs.pop("return_result","field")
        return_shape = kwargs.pop("return_shape","full")
        free_prop = kwargs.pop("free_prop",False)

        free_prop = free_prop or (self.dn is None)

        if return_val=="field":
            res_type = Bpm3d._complex_type
        elif return_val=="intens":
            res_type = Bpm3d._real_type
        else:
            raise ValueError()

        if not return_shape in ["last", "full"]:
            raise ValueError()


        if u0 is None:
            u0 = self.u0_plane()


        Nx, Ny, Nz = self.shape

        assert offset>=0 and offset<(Nz-1)

        if return_shape=="full":
            u = OCLArray.empty((Nz-offset,Ny,Nx),dtype=res_type)


        self._buf_plane.write_array(u0)

        #copy the first plane
        if return_shape=="full":
            if self._is_subsampled:
                self._img_xy.copy_buffer(self._buf_plane)
                self._copy_down_img(self._img_xy,u,0)
            else:
                self._copy_down_buf(self._buf_plane,u,0)

        for i in xrange(Nz-1-offset):
            for j in xrange(self.simul_z):

                fft(self._buf_plane, inplace = True, plan  = self._plan)
                self._mult_complex(self._buf_plane, self._buf_H)
                fft(self._buf_plane, inplace = True, inverse = True, plan  = self._plan)
                if not free_prop:
                    self._mult_dn(self._buf_plane,(i+offset+(j+1.)/self.simul_z))

            if return_shape=="full":
                if self._is_subsampled and self.simul_xy!=self.shape[:2]:
                    self._img_xy.copy_buffer(self._buf_plane)
                    self._copy_down_img(self._img_xy,u,(i+1)*(Nx*Ny))
                else:
                    self._copy_down_buf(self._buf_plane,u,(i+1)*(Nx*Ny))

        if return_shape=="full":
            return u.get()
        else:
            return self._buf_plane


    def __repr__(self):
        return "Bpm3d class \nsize: \t%s\nshape:\t %s\nresolution: (%.4f,%.4f,%.4f)  "%(self.size,self.shape,self.dx,self.dy,self.dz)


if __name__ == '__main__':

    from time import time

    shape = (256,256,256)
    size = (40,40,40)
    #size = (200,200,200)


    #
    # m1 = Bpm3d(shape = shape, size = size, simul_z=2,fftplan_kwargs={"fast_math":True})#simul_z = 1, simul_xy=(512,512))
    # u1 = m1._propagate(m1.u0_beam(NA = .4))
    # m2 = Bpm3d(shape = shape, size = size,simul_z = 2, fftplan_kwargs={"fast_math":False})#, simul_xy=(512,512))
    # u2 = m2._propagate(m2.u0_beam(NA = .4))


    # m1 = Bpm3d(shape = shape, size = size, fftplan_kwargs={"fast_math":False})#simul_z = 1, simul_xy=(512,512))
    # m1._buf_plane.write_array(m1.u0_beam(NA = .3))
    #
    # m2 = Bpm3d(shape = shape, size = size)#simul_z = 1, simul_xy=(512,512))
    # m2._buf_plane.write_array(m2.u0_beam(NA = .3))
    # m2.dz *= 2
    # m2._fill_propagator(1.)
    #
    # #m1._buf_plane.write_array(m1.u0_plane())
    # for i in range(100):
    #     fft(m1._buf_plane, inplace = True, plan  = m1._plan)
    #     #m1._mult_complex(m1._buf_plane, m1._buf_H)
    #     fft(m1._buf_plane, inplace = True, inverse = True, plan  = m1._plan)
    #     # fft(m1._buf_plane, inplace = True, plan  = m1._plan)
    #     # #m1._mult_complex(m1._buf_plane, m1._buf_H)
    #     # fft(m1._buf_plane, inplace = True, inverse = True, plan  = m1._plan)
    #     #
    #     a1 = m1._buf_plane.get()
    #     print np.amax(np.abs(a1-m1.u0_beam(NA = .3)))/np.amax(abs(a1))
    #
    #     # fft(m2._buf_plane, inplace = True, plan  = m2._plan)
    #     # #m2._mult_complex(m2._buf_plane, m2._buf_H)
    #     # fft(m2._buf_plane, inplace = True, inverse = True, plan  = m2._plan)
    #     #
    #     # a2 = m2._buf_plane.get()
    #     #
    #     # print np.amax(np.abs(a1-a2))/np.amax(abs(a1))
    #
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

    # # m = Bpm3d(size = size, dn = dn,simul_z = 1, simul_xy=(512,512))
    # #
    # #
    # # # m = Bpm3d(size = size, dn = dn,simul_z = 2, simul_xy=(1024,)*2)
    # #
    # #
    # # t = time()
    # #
    # # #u = m._propagate( return_val="intens")
    # # #u = m._propagate(m.u0_beam(NA = .4), return_result="intens")
    # # u = m._propagate(m.u0_beam(NA = .4), return_result = "intens")
    # # #u = m._propagate(np.roll(m.u0_cylindrical(NA = .2),50,0), return_result = "intens")
    # #
    # #
    # # print time()-t
    # #
    # #
