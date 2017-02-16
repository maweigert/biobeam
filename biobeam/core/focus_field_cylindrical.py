# coding=utf-8
"""
PSF calulcations for cylidrical lenses

see e.g.
Purnapatra, Subhajit B. Mondal, Partha P.
Determination of electric field at and near the focus of a cylindrical lens for applications in fluorescence microscopy (2013)
"""

from __future__ import absolute_import
from __future__ import print_function
from gputools import OCLArray, OCLImage, OCLProgram
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


def focus_field_cylindrical(shape=(128, 128, 128),
                            units=(0.1, 0.1, 0.1),
                            lam=.5,
                            NA=.3,
                            n0=1.,
                            return_all_fields=False,
                            n_integration_steps=100):
    """calculates the focus field for a perfect, aberration free cylindrical lens after
    x polarized illumination via the vectorial debye diffraction integral (see [2]_).
    The pupil function is given by the numerical aperture NA



    Parameters
    ----------

    shape: Nx,Ny,Nz
        the shape of the geometry
    units: dx,dy,dz
        the pixel sizes in microns
    lam: float
        the wavelength of light used in microns
    NA: float
        the numerical aperture of the lens
    n0: float
        the refractive index of the medium
    return_all_fields: boolean
        if True, returns u,ex,ey,ez where ex/ey/ez are the complex field components
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

    >>> u, ex, ey, ez = focus_field_cylindrical((128,128,128), (0.1,0.1,.1), lam=.5, NA = .4, return_all_field=True)

    References
    ----------

    .. [2] Colin J. R. Sheppard: Cylindrical lenses—focusing and imaging: a review, Appl. Opt. 52, 538-545 (2013)


    """

    p = OCLProgram(absPath("kernels/psf_cylindrical.cl"),
                   build_options=["-I", absPath("kernels"), "-D", "INT_STEPS=%s"%n_integration_steps])

    Nx, Ny, Nz = shape
    dx, dy, dz = units

    alpha = np.arcsin(NA/n0)

    u_g = OCLArray.empty((Nz, Ny), np.float32)
    ex_g = OCLArray.empty((Nz, Ny), np.complex64)
    ey_g = OCLArray.empty((Nz, Ny), np.complex64)
    ez_g = OCLArray.empty((Nz, Ny), np.complex64)

    t = time.time()

    p.run_kernel("psf_cylindrical", u_g.shape[::-1], None,
                 ex_g.data,
                 ey_g.data,
                 ez_g.data,
                 u_g.data,
                 np.float32(-dy*(Ny//2)), np.float32((Ny-1-Ny//2)*dy),
                 np.float32(-dz*(Nz//2)), np.float32((Nz-1-Nz//2)*dz),
                 np.float32(lam/n0),
                 np.float32(alpha))

    u = np.array(np.repeat(u_g.get()[..., np.newaxis], Nx, axis=-1))
    ex = np.array(np.repeat(ex_g.get()[..., np.newaxis], Nx, axis=-1))
    ey = np.array(np.repeat(ey_g.get()[..., np.newaxis], Nx, axis=-1))
    ez = np.array(np.repeat(ez_g.get()[..., np.newaxis], Nx, axis=-1))

    print("time in secs:", time.time()-t)

    if return_all_fields:
        return u, ex, ey, ez
    else:
        return u


def focus_field_cylindrical_plane(shape=(128, 128),
                                  units=(.1, .1),
                                  z=0.,
                                  lam=.5,
                                  NA=.3, n0=1.,
                                  ex_g=None,
                                  n_integration_steps=200):
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
    >>> ex = focus_field_cylindrical_plane((256,256), (0.1,0.1), z = 4., lam=.5, NA = (.4,.5))

    See Also
    --------
    biobeam.focus_field_cylindrical : the 3d function


    """

    p = OCLProgram(absPath("kernels/psf_cylindrical.cl"),
                   build_options=["-I", absPath("kernels"), "-D", "INT_STEPS=%s"%n_integration_steps])

    Nx, Ny = shape
    dx, dy = units

    alpha = np.arcsin(NA/n0)

    if ex_g is None:
        use_buffer = False
        ex_g = OCLArray.empty((Ny, Nx), np.complex64)
    else:
        use_buffer = True

    assert ex_g.shape[::-1]==shape

    p.run_kernel("psf_cylindrical_plane", (Nx, Ny), None,
                 ex_g.data,
                 np.float32(-dy*(Ny//2)), np.float32((Ny-1-Ny//2)*dy),
                 np.float32(-z),
                 np.float32(lam/n0),
                 np.float32(alpha))

    if not use_buffer:
        return ex_g.get()


if __name__=='__main__':
    u = focus_field_cylindrical((128,)*3, (.1,)*3, NA=.3)


    # ex2  = focus_field_cylindrical_plane((128,)*2,(.1,)*2,z = -6.4, NA = .01)
