"""
focus field (psf) calulcations for a single beam originating from objectives
with ring like (gaussian) and annulus (bessel ) pupil functions

via the Debye Wolf integral


see e.g.
Foreman, M. R., & Toeroek, P. (2011). Computational methods in vectorial imaging.
Journal of Modern Optics, 58(5-6)
"""




from gputools import OCLArray, OCLImage, OCLProgram
import itertools
import numpy as np
import time
import os
import sys

def absPath(myPath):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        return os.path.join(base_path, os.path.basename(myPath))
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, myPath)


def focus_field_beam(shape, units, lam =.5, NA = .6, n0 = 1.,
                     return_all_fields = False,
                     n_integration_steps = 200):
    """calculates the focus field for a perfect, aberration free optical system for
    x polzarized illumination via the vectorial debye diffraction integral (see [1]_).
    The pupil function is given by numerical aperture(s) NA (that can be a list to
    model bessel beams, see further below)


    Parameters
    ----------

    shape: Nx,Ny,Nz
        the shape of the geometry
    units: dx,dy,dz
        the pixel sizes in microns
    lam: float
        the wavelength of light used in microns
    NA: float/list
        the numerical aperture(s) of the illumination objective
        that is either a single number (for gaussian beams) or an
        even length list of NAs (for bessel beams), e.g.
        `NA = [0.5,0.55]` lets light through the annulus 0.5<0.55 (making a bessel beam ) or
        `NA = [0.1,0.2,0.5,0.6]` lets light through the annulus 0.1<0.2 and 0.5<0.6 making a
        beating double bessel beam...
    n0: float
        the refractive index of the medium
    n_integration_steps: int
        number of integration steps to perform
    return_all_fields: boolean
        if True returns also the complex vectorial field components

    Returns
    -------
    u: ndarray
        the intensity of the focus field
    (u,ex,ey,ez): list(ndarray)
        the intensity of the focus field and the complex field components (if return_all_fields is True)



    Example
    -------

    >>> u = focus_field_beam((128,128,128), (0.1,0.1,.1), lam=.5, NA = .4)

    References
    ----------

    .. [1] Matthew R. Foreman, Peter Toeroek, *Computational methods in vectorial imaging*, Journal of Modern Optics, 2011, 58, 5-6, 339

    """


    p = OCLProgram(absPath("kernels/psf_debye.cl"),
            build_options = ["-I",absPath("kernels"),"-D","INT_STEPS=%s"%n_integration_steps])

    if np.isscalar(NA):
        NA = [0.,NA]
    
    Nx0, Ny0, Nz0 = shape
    dx, dy, dz = units

    #FIXME: the loop below does not yet work for odd inputs
    if not Nx0%2+Ny0%2+Nz0%2==0:
        raise NotImplementedError("odd shapes not supported yet")


    alphas = np.arcsin(np.array(NA)/n0)
    assert len(alphas)%2 ==0

    # as we assume the psf to be symmetric, we just have to calculate each octant
    Nx = Nx0/2+1
    Ny = Ny0/2+1
    Nz = Nz0/2+1

    u_g = OCLArray.empty((Nz,Ny,Nx),np.float32)
    ex_g = OCLArray.empty(u_g.shape,np.complex64)
    ey_g = OCLArray.empty(u_g.shape,np.complex64)
    ez_g = OCLArray.empty(u_g.shape,np.complex64)

    alpha_g = OCLArray.from_array(alphas.astype(np.float32))

    t = time.time()
    
    p.run_kernel("debye_wolf",u_g.shape[::-1],None,
                 ex_g.data,ey_g.data,ez_g.data, u_g.data,
                 np.float32(1.),np.float32(0.),
                 np.float32(0.),np.float32(dx*(Nx-1.)),
                 np.float32(0.),np.float32(dy*(Ny-1.)),
                 np.float32(0.),np.float32(dz*(Nz-1.)),
                 np.float32(1.*lam/n0),
                 alpha_g.data, np.int32(len(alphas)))

    u = u_g.get()
    ex = ex_g.get()
    ey = ey_g.get()
    ez = ez_g.get()

    u_all = np.empty((Nz0,Ny0,Nx0),np.float32)
    ex_all = np.empty((Nz0,Ny0,Nx0),np.complex64)
    ey_all = np.empty((Nz0,Ny0,Nx0),np.complex64)
    ez_all = np.empty((Nz0,Ny0,Nx0),np.complex64)

    sx = [slice(0,Nx),slice(Nx,Nx0)]
    sy = [slice(0,Ny),slice(Ny,Ny0)]
    sz = [slice(0,Nz),slice(Nz,Nz0)]



    # spreading the calculated octant to the full volume
    for i,j,k in itertools.product([0,1],[0,1],[0,1]):

        # i, j, k = 0 indicates the + octant

        u_all[sz[1-i],sy[1-j],sx[1-k]] = u[1-i:Nz-1+i,1-j :Ny-1+j,1-k :Nx-1+k][::(-1)**i,::(-1)**j,::(-1)**k]
        if i ==0:
            ex_all[sz[1-i],sy[1-j],sx[1-k]] = ex[1-i:Nz-1+i,1-j :Ny-1+j,1-k :Nx-1+k][::(-1)**i,::(-1)**j,::(-1)**k]
            ey_all[sz[1-i],sy[1-j],sx[1-k]] = ey[1-i:Nz-1+i,1-j :Ny-1+j,1-k :Nx-1+k][::(-1)**i,::(-1)**j,::(-1)**k]
            ez_all[sz[1-i],sy[1-j],sx[1-k]] = ez[1-i:Nz-1+i,1-j :Ny-1+j,1-k :Nx-1+k][::(-1)**i,::(-1)**j,::(-1)**k]

        else:
            ex_all[sz[1-i],sy[1-j],sx[1-k]] = np.conjugate(ex[1-i:Nz-1+i,1-j :Ny-1+j,1-k :Nx-1+k][::(-1)**i,::(-1)**j,::(-1)**k])
            ey_all[sz[1-i],sy[1-j],sx[1-k]] = np.conjugate(ey[1-i:Nz-1+i,1-j :Ny-1+j,1-k :Nx-1+k][::(-1)**i,::(-1)**j,::(-1)**k])
            ez_all[sz[1-i],sy[1-j],sx[1-k]] = np.conjugate(ez[1-i:Nz-1+i,1-j :Ny-1+j,1-k :Nx-1+k][::(-1)**i,::(-1)**j,::(-1)**k])

    if return_all_fields:
        return u_all, ex_all, ey_all, ez_all
    else:
        return u_all


def focus_field_beam_plane(shape = (128,128),
                            units = (.1,.1),
                            z = 0.,
                            lam = .5, NA = .6, n0 = 1.,
                            ex_g = None,
                            n_integration_steps = 200):
    """calculates the complex 2d input field at position -z of a \
     perfect, aberration free optical system



    Parameters
    ----------
    shape: Nx,Ny
        the 2d shape of the geometry
    units: dx,dy
        the pixel sizes in microns
    z:  float
        defocus position in microns, such that the beam would focus at z
        e.g. an input field with z = 10. would hav its focus spot after 10 microns
    lam: float
        the wavelength of light used in microns
    NA: float/list
        the numerical aperture(s) of the illumination objective
        that is either a single number (for gaussian beams) or an
        even length list of NAs (for bessel beams), e.g.
        NA = [0.5,0.55] lets light through the annulus 0.5<0.55 (making a bessel beam ) or
        NA = [0.1,0.2,0.5,0.6] lets light through the annulus 0.1<0.2 and 0.5<0.6 making a
        beating double bessel beam...
    n0: float
        the refractive index of the medium
    n_integration_steps: int
        number of integration steps to perform

    Returns
    -------
    ex: ndarray
        the complex field

    Example
    -------

    >>> # the input pattern of a bessel beam that will focus after 4 microns
    >>> ex = focus_field_beam_plane((256,256), (0.1,0.1), z = 4, lam=.5, NA = (.4,.5))



    See Also
    --------
    biobeam.focus_field_beam : the 3d function


    """


    p = OCLProgram(absPath("kernels/psf_debye.cl"),
            build_options = ["-I",absPath("kernels"),"-D","INT_STEPS=%s"%n_integration_steps])

    if np.isscalar(NA):
        NA = [0.,NA]

    Nx, Ny = shape
    dx, dy = units

    alphas = np.arcsin(np.array(NA)/n0)
    assert len(alphas)%2 ==0

    if ex_g is None:
        use_buffer = False
        ex_g = OCLArray.empty((Ny,Nx),np.complex64)
    else:
        use_buffer = True

    assert ex_g.shape[::-1] == shape

    alpha_g = OCLArray.from_array(alphas.astype(np.float32))

    t = time.time()

    p.run_kernel("debye_wolf_plane",(Nx,Ny),None,
                 ex_g.data,
                 np.float32(1.),np.float32(0.),
                 np.float32(-(Nx/2)*dx),np.float32((Nx-Nx/2)*dx),
                 np.float32(-(Ny/2)*dy),np.float32((Ny-Ny/2)*dy),
                 np.float32(z),
                 np.float32(lam/n0),
                 alpha_g.data, np.int32(len(alphas)))

    print "time in secs:" , time.time()-t

    if not use_buffer:
        return ex_g.get()




def test_bessel(n,x):
    x_g = OCLArray.from_array(x.astype(float32))
    res_g = OCLArray.empty_like(x.astype(float32))

    p = OCLProgram(absPath("kernels/bessel.cl"))
    p.run_kernel("bessel_fill",x_g.shape,None,
                 x_g.data,res_g.data,int32(n))

    return res_g.get()


    
def test_debye():
    
    lam = .5
    NA1 = .7
    NA2 = .76

    Nx = 128
    Nz = 128 
    dx = .05
    
    u,ex,ey,ez = focus_field_beam((Nx, Nx, Nz), (dx, dx, dx),
                                  lam = lam, NAs = [0.89,0.9], n_integration_steps= 100)

    u2,ex2,ey2,ez2 = focus_field_beam_new((Nx, Nx, Nz), (dx, dx, dx),
                                           lam = lam, NAs = [0.89,0.9], n_integration_steps= 100)
    
    return u,u2

if __name__ == '__main__':

    ex_p = focus_field_beam_plane((128,)*2,(.1,)*2,
                                   z = -6.4, NA = .3, n0 = 1.)

    u1 = focus_field_beam((128,)*3, (0.1,)*3, lam=.5, NA = .4)

