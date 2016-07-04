"""
the main class for gpu accelerated bpm propagation

mweigert@mpi-cbg.de

"""

import numpy as np
from gputools import OCLArray, OCLImage, OCLProgram, get_device
from gputools import fft, fft_plan

#from gputools import OCLReductionKernel


from biobeam.focus_field import focus_field_cylindrical_plane, \
    focus_field_beam_plane, \
    focus_field_lattice_plane




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

    def __init__(self, size = None,
                 shape = None,
                 units = None,
                 dn = None,
                 lam = .5,
                 n0 = 1.,
                 simul_xy = None,
                 simul_z =1,
                 n_volumes = 1,
                 enforce_subsampled = False,
                 fftplan_kwargs = {}):
        """

        Parameters
        ----------
        size: (Sx,Sy,Sz)
            the size of the geometry in microns (Sx,Sy,Sz)
        shape: (Nx,Ny,Nz)
            the shape of the geometry in pixels (Nx,Ny,Nz)
        units: (dx,dy,dz)
            the voxelsizes in microns (dx,dy,dz)
        dn: ndarray (float32|complex64)
            refractive index distribution, dn.shape != (Nz,Ny,Nx)
        lam: float
            the wavelength in microns
        n0: float
            the refractive index of the surrounding media
        simul_xy: (Nx,Ny,Nz), optional
            the shape of the 2d computational geometry in pixels (Nx,Ny)
            (e.g. subsampling in xy)
        simul_z: int, optional
            the subsampling factor along z
        n_volumes: int
            splits the domain into chunks if GPU memory is not
            large enough (will be set automatically)

        Example
        -------

        >>> m = Bpm3d(size = (10,10,10),shape = (256,256,256),units = (0.1,0.1,0.1),lam = 0.488,n0 = 1.33)

        """

        if shape is None and dn is None:
            raise ValueError("either shape or dn have to be given!")
        if not(shape is None or dn is None) and dn.shape != shape[::-1]:
            raise ValueError("shape != dn.shape")
        if not (size is None or units is None):
            raise ValueError("either give size or units but not both!")

        if not units is None:
            size = [(s-1.)*u for s, u in zip(shape, units)]


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

            #FIXME: this is still stupid
            if not self.dn is None:
                self.dn_mean = np.mean(np.real(self.dn),axis=(1,2))


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

    def _mult_dn(self, buf, zPos, dn0 = 0.):
        if (self._is_subsampled and self.dn.dtype == Bpm3d._complex_type) or\
            (not self._is_subsampled and self._buf_dn.dtype == Bpm3d._complex_type):
            self._mult_dn_complex(buf,zPos, dn0)
        else:
            self._mult_dn_float(buf,zPos, dn0)


    def _mult_dn_float(self, buf, zPos, dn0):
        if self._is_subsampled:
            self._kernel_mult_dn_img_float(self._queue, self.simul_xy, None,
                                    buf.data,self._im_dn,
                                    np.float32(self.k0*self.dz),
                                           np.float32(dn0),
                                    np.float32(zPos/(self.shape[-1]-1.))
                                           )
        else:
            Nx, Ny = self.shape[:2]
            self._kernel_mult_dn_buf_float(self._queue, self.shape[:2], None,
                                    buf.data,self._buf_dn.data,
                                    np.float32(self.k0*self.dz),
                                           np.float32(dn0),
                                    np.int32(zPos*Nx*Ny))

    def _mult_dn_complex(self, buf, zPos, dn0):
        if self._is_subsampled:
            self._kernel_mult_dn_img_complex(self._queue, self.simul_xy, None,
                                    buf.data,self._im_dn,
                                    np.float32(self.k0*self.dz),
                                             np.float32(dn0),
                                    np.float32(zPos/(self.shape[-1]-1.)))

        else:
            Nx, Ny = self.shape[:2]
            self._kernel_mult_dn_buf_complex(self._queue, self.shape[:2], None,
                                    buf.data,self._buf_dn.data,
                                    np.float32(self.k0*self.dz),
                                             np.float32(dn0),
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


    def u0_beam(self, center = (0,0),zfoc = None, NA = .3, n_integration_steps = 200):
        if zfoc is None:
            zfoc = .5*self.size[-1]


        u0 =  focus_field_beam_plane(shape = self.simul_xy, units = (self.dx, self.dy),
                      z = zfoc,NA = NA,
                      lam = self.lam, n0 = self.n0,
                     n_integration_steps=n_integration_steps)
        cx,cy = center
        return np.roll(np.roll(u0,cy,0),cx,1)

    def u0_cylindrical(self, center = (0,0),zfoc =None ,  NA = .3):
        if zfoc is None:
            zfoc = .5*self.size[-1]

        u0 = focus_field_cylindrical_plane(shape = self.simul_xy, units = (self.dx, self.dy),
                      z = zfoc, NA = NA,
                      lam = self.lam, n0 = self.n0).conjugate()
        cx,cy = center
        return np.roll(np.roll(u0,cy,0),cx,1)


    def u0_lattice(self, center = (0,0),zfoc =None ,  NA1 = .3, NA2 = .4,
                   sigma = .1):

        if zfoc is None:
            zfoc = .5*self.size[-1]

        u0 = focus_field_lattice_plane(shape = self.simul_xy,
                            units = (self.dx, self.dy),
                      z = zfoc, NA1 = NA1, NA2 = NA2,
                            sigma = sigma,
                      lam = self.lam, n0 = self.n0)
        cx,cy = center
        return np.roll(np.roll(u0,cy,0),cx,1)



    def _propagate(self, u0 = None, offset = 0,
                   return_comp = "field",
                   return_shape = "full",
                   free_prop = False,
                   slow_mean = False,
                   **kwargs):
        """

        kwargs:
            return_comp in ["field", "intens"]
            return_shape in ["last", "full"]
            free_prop = False|True
        """

        # return_val = kwargs.pop("return_result","field")
        # return_shape = kwargs.pop("return_shape","full")
        # free_prop = kwargs.pop("free_prop",False)

        free_prop = free_prop or (self.dn is None)

        if return_comp=="field":
            res_type = Bpm3d._complex_type
        elif return_comp=="intens":
            res_type = Bpm3d._real_type
        else:
            raise ValueError(return_comp)

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



        dn0 = 0
        for i in xrange(Nz-1-offset):
            if not self.dn is None and not free_prop:
                if slow_mean:
                    if return_shape=="full":
                        raise NotImplementedError()
                    else:
                        tmp = OCLArray.empty((1,Ny,Nx),dtype=res_type)
                        if self._is_subsampled:
                            self._img_xy.copy_buffer(self._buf_plane)
                            self._copy_down_img(self._img_xy,tmp,0)
                        else:
                            self._copy_down_buf(self._buf_plane,tmp,0)

                        dn0 = np.sum(np.abs(self.dn[i])*tmp.get())/np.sum(np.abs(self.dn[i])+1.e-10)
                        print dn0
                        self._fill_propagator(self.n0+dn0)
                else:
                    if self.dn_mean[i+offset] != dn0:
                        dn0 = self.dn_mean[i+offset]
                        self._fill_propagator(self.n0+dn0)




            for j in xrange(self.simul_z):

                fft(self._buf_plane, inplace = True, plan  = self._plan)
                self._mult_complex(self._buf_plane, self._buf_H)
                fft(self._buf_plane, inplace = True, inverse = True, plan  = self._plan)
                if not free_prop:
                    self._mult_dn(self._buf_plane,(i+offset+(j+1.)/self.simul_z),dn0)

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

    def _aberr_from_field(self,u0,NA, n_zern = 20):
        """assume NA is the same"""
        from phasediv import aberr_from_field, PhaseDiv2
        if not hasattr(self,"_PD2") or self._PD2.NA != NA:
            self._PD2 = PhaseDiv2(self.simul_xy[::-1],(self.dy,self.dx), NA = NA, n=self.n0)
            self._NA = NA
        assert self._NA ==NA
        return aberr_from_field(u0,units=(self.dx,self.dy),
                                lam = self.lam,NA=NA,n=self.n0,
                                pd_obj = self._PD2,
                                n_zern = n_zern)

    def _aberr_from_field_NA(self,u0,NA, n_zern = 20):
        from phasediv import aberr_from_field
        return aberr_from_field(u0,units=(self.dy,self.dx),
                                lam = self.lam,NA=NA,n=self.n0,
                                n_zern = n_zern)

    def aberr_at(self,NA = .4, center = (0,0,0), n_zern = 20,
                 n_integration_steps = 200):
        """c = (cx,cy,cz) in realtive pixel coordinates wrt the center

        returns phi, zern
        """


        cx,cy,cz = center

        offset = self.shape[-1]/2+1+cz


        u0 = np.roll(np.roll(self.u0_beam(zfoc = 0.,NA=NA,
                                          n_integration_steps = n_integration_steps),cy,0),cx,1)

        u_forth = self._propagate(u0 = u0,offset=offset, return_shape="last").get()

        u_back = self._propagate(u0 = u_forth.conjugate(),free_prop = True,
                                 offset=offset, return_shape="last").get()
        u_back = np.roll(np.roll(u_back,-cy,0),-cx,1)

        self._u_back = u_back
        return self._aberr_from_field(u_back,NA=NA)


    def aberr_field_grid(self,NA , cxs, cys , cz, n_zern = 20,
                 n_integration_steps = 200):
        """
        cxs, cys are equally spaced 1d arrays defining the grid
        """


        CYs, CXs = np.meshgrid(cys,cxs,indexing = "ij")
        CYs, CXs = CYs.flatten(),CXs.flatten()

        Npad = int(np.floor(min(abs(cxs[1]-cxs[0]),abs(cys[1]-cys[0]))/32)*16)
        assert Npad>1

        Nslice0 = (slice(self.simul_xy[1]/2-Npad,self.simul_xy[1]/2+Npad),
                  slice(self.simul_xy[0]/2-Npad,self.simul_xy[0]/2+Npad))

        Nslices = [(slice(self.simul_xy[1]/2-Npad+cy,self.simul_xy[1]/2+Npad+cy),
                  slice(self.simul_xy[0]/2-Npad+cx,self.simul_xy[0]/2+Npad+cx)) for cx,cy in zip(CXs,CYs)]

        offset = self.shape[-1]/2+1+cz

        u0_single = self.u0_beam(zfoc = 0.,NA=NA,
                                          n_integration_steps = n_integration_steps)

        u0 = np.zeros_like(u0_single)
        for cx,cy, nslice in zip(CXs,CYs, Nslices):
            u0[nslice] += u0_single[Nslice0]

        #u0 = reduce(np.add,[np.roll(np.roll(u0_single,cy,0),cx,1) for cx,cy in zip(Xs,Ys)])

        print "propagation"
        u_forth = self._propagate(u0 = u0,offset=offset, return_shape="last").get()

        u_back = self._propagate(u0 = u_forth.conjugate(),free_prop = True,
                                 offset=offset, return_shape="last").get()

        self._u_back = u_back

        # get the phases


        us = [u_back[nslice] for nslice in Nslices]

        print "setup"
        from phasediv import aberr_from_field, PhaseDiv2
        self._PD2_pad = PhaseDiv2(us[0].shape,(self.dy,self.dx), NA = NA, n=self.n0)




        phis = []
        zerns = []
        print "getting aberrations"
        for i,u0 in enumerate(us):
            print "%s/%s"%(i+1,len(us))
            p,z = aberr_from_field(u0,units=(self.dx,self.dy),
                                lam = self.lam,NA=NA,n=self.n0,
                                pd_obj = self._PD2_pad,
                                n_zern = n_zern)
            phis.append(p)
            zerns.append(z)
        return np.array(phis),np.array(zerns)


    def __repr__(self):
        return "Bpm3d class \nsize: \t%s\nshape:\t %s\nresolution: (%.4f,%.4f,%.4f)  "%(self.size,self.shape,self.dx,self.dy,self.dz)




if __name__ == '__main__':

    from time import time

    shape = (256,256,256)
    size = (40,40,40)
    shape = (512,256,256)
    size = (80.16,40,40)
    #shape = (256,512,256)
    # size = (40,80,40)

    if not "dn" in locals():
        xs = [np.linspace(-.5*s,.5*s,N) for s,N in zip(size,shape)]
        Xs = np.meshgrid(*xs[::-1],indexing="ij")
        R = np.sqrt(reduce(np.add,[_X**2 for _X in Xs]))
        dn = (.1-.001j)*(R<size[-1]/5)

    m = Bpm3d(size = size, dn = dn)


    phi, zern = m.aberr_at(center = (0,0,0))
