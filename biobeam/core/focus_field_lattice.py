"""
PSF calulcations for cylidrical lenses

see e.g.
Purnapatra, Subhajit B. Mondal, Partha P.
Determination of electric field at and near the focus of a cylindrical lens for applications in fluorescence microscopy (2013)
"""

from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import numpy as np
import time
from gputools import OCLArray, OCLProgram


def absPath(myPath):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        return os.path.join(base_path, os.path.basename(myPath))
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(base_path, myPath)


def _poly_points(N=6):
    """returns the coordinates of a regular polygon on the unit circle"""
    ts = np.pi*(.5+2./N*np.arange(N))
    return np.stack([np.cos(ts), np.sin(ts)])


def focus_field_lattice(shape=(128, 128, 128),
                        units=(0.1, 0.1, 0.1),
                        lam=.5, NA1=.4, NA2=.5,
                        sigma=.1,
                        kpoints=6,
                        return_all_fields=False,
                        n0=1., n_integration_steps=100):
    """Calculates the focus field for a bessel lattice.
    The pupil function consists out of discrete points (kpoints) superimposed on an annulus (NA1<NA2)
    which are smeared out by a 1d gaussian of given sigma creating an array of bessel beams in the
    focal plane (see [3]_ ).


    Parameters
    ----------

    shape: Nx,Ny,Nz
        the shape of the geometry
    units: dx,dy,dz
        the pixel sizes in microns
    lam: float
        the wavelength of light used in microns
    NA1: float/list
        the numerical aperture of the inner ring
    NA2: float/list
        the numerical aperture of the outer ring
    sigma: float
        the standard deviation of the gaussian smear function applied to each point on the aperture
        (the bigger sigma, the tighter the sheet in y)
    kpoints: int/ (2,N) array
        defines the set of points on the aperture that create the lattice, can be
        - a (2,N) ndarray, such that kpoints[:,i] are the coordinates of the ith point
        - a single int, defining points on a regular polygon (e.g. 4 for a square lattice, 6 for a hex lattice)
        :math:`k_i = \\arcsin\\frac{NA_1+NA_2}{2 n_0} \\begin{pmatrix} \\cos \\phi_i \\\\ \\sin \\phi_i \\end{pmatrix}\quad, \\phi_i = \\frac{\\pi}{2}+\\frac{2i}{N}`
        
    n0: float
        the refractive index of the medium
    n_integration_steps: int
        number of integration steps to perform
    return_all_fields: boolean
        if True, returns u,ex,ey,ez where ex/ey/ez are the complex vector field components

    Returns
    -------
    u: ndarray
        the intensity of the focus field
    (u,ex,ey,ez): list(ndarray)
        the intensity of the focus field and the complex field components (if return_all_fields is True)

    Example
    -------

    >>> u = focus_field_lattice((128,128,128), (0.1,0.1,.1), lam=.5, NA1 = .44, NA2 = .55, kpoints = 6)

    References
    ----------

    .. [3] Chen et al. Lattice light-sheet microscopy: imaging molecules to embryos at high spatiotemporal resolution. Science 346, (2014).


    """

    alpha1 = np.arcsin(1.*NA1/n0)
    alpha2 = np.arcsin(1.*NA2/n0)

    if np.isscalar(kpoints):
        kxs, kys = np.arcsin(.5*(NA1+NA2)/n0)*_poly_points(kpoints)
    else:
        kxs, kys = 1.*kpoints/n0

    p = OCLProgram(absPath("kernels/psf_lattice.cl"),
                   build_options=["-I", absPath("kernels"), "-D", "INT_STEPS=%s"%n_integration_steps])

    kxs = np.array(kxs)
    kys = np.array(kys)

    Nx, Ny, Nz = shape
    dx, dy, dz = units

    u_g = OCLArray.empty((Nz, Ny, Nx), np.float32)
    ex_g = OCLArray.empty((Nz, Ny, Nx), np.complex64)
    ey_g = OCLArray.empty((Nz, Ny, Nx), np.complex64)
    ez_g = OCLArray.empty((Nz, Ny, Nx), np.complex64)

    kxs_g = OCLArray.from_array(kxs.astype(np.float32))
    kys_g = OCLArray.from_array(kys.astype(np.float32))

    t = time.time()

    p.run_kernel("debye_wolf_lattice", (Nx, Ny, Nz),
                 None,
                 ex_g.data,
                 ey_g.data,
                 ez_g.data,
                 u_g.data,
                 np.float32(1.), np.float32(0.),
                 # np.float32(-dx*(Nx-1)//2.),np.float32(dx*(Nx-1)//2.),
                 # np.float32(-dy*(Ny-1)//2.),np.float32(dy*(Ny-1)//2.),
                 # np.float32(-dz*(Nz-1)//2.),np.float32(dz*(Nz-1)//2.),
                 np.float32(dx*(-Nx//2)), np.float32(dx*(Nx//2-1)),
                 np.float32(dy*(-Ny//2)), np.float32(dy*(Ny//2-1)),
                 np.float32(dz*(-Nz//2)), np.float32(dz*(Nz//2-1)),
                 np.float32(1.*lam/n0),
                 np.float32(alpha1),
                 np.float32(alpha2),
                 kxs_g.data,
                 kys_g.data,
                 np.int32(len(kxs)),
                 np.float32(sigma)
                 )

    u = u_g.get()

    if return_all_fields:
        ex = ex_g.get()
        ey = ey_g.get()
        ez = ez_g.get()
        return u, ex, ey, ez
    else:
        return u


def focus_field_lattice2(shape=(128, 128, 128),
                         units=(0.1, 0.1, 0.1),
                         lam=.5, NA1=.4, NA2=.5,
                         sigma=.1,
                         kpoints=6,
                         n0=1., n_integration_steps=100):
    """

    kpoints can be
      - either a (2,N) dimensional array such that
       kpoints[:,i] are the coordinates of the ith lattice point in back pupil coordinates
      - a single number, e.g. kpoints = 6, where kpoints are then assumed to lie on regular
        kpoints-polygon, i.e.
            kpoints = .5*(NA1+NA2)*np.pi*(.5+2./N*arange(N))

    """

    alpha1 = np.arcsin(NA1/n0)
    alpha2 = np.arcsin(NA2/n0)

    if np.isscalar(kpoints):
        kxs, kys = np.arcsin(.5*(NA1+NA2)/n0)*_poly_points(kpoints)
    else:
        kxs, kys = kpoints

    p = OCLProgram(absPath("kernels/psf_lattice.cl"),
                   build_options=["-I", absPath("kernels"), "-D", "INT_STEPS=%s"%n_integration_steps])

    kxs = np.array(kxs)
    kys = np.array(kys)

    Nx, Ny, Nz0 = shape
    dx, dy, dz = units

    # the psf is symmetric in z, we just have to calculate one half plane
    Nz = Nz0//2+1

    u_g = OCLArray.empty((Nz, Ny, Nx), np.float32)
    ex_g = OCLArray.empty((Nz, Ny, Nx), np.complex64)
    ey_g = OCLArray.empty((Nz, Ny, Nx), np.complex64)
    ez_g = OCLArray.empty((Nz, Ny, Nx), np.complex64)
    kxs_g = OCLArray.from_array(kxs.astype(np.float32))
    kys_g = OCLArray.from_array(kys.astype(np.float32))

    t = time.time()

    p.run_kernel("debye_wolf_lattice", (Nx, Ny, Nz),
                 None,
                 ex_g.data,
                 ey_g.data,
                 ez_g.data,
                 u_g.data,
                 np.float32(1.), np.float32(0.),
                 np.float32(dx*(-Nx//2)), np.float32(dx*(Nx//2-1)),
                 np.float32(dy*(-Ny//2)), np.float32(dy*(Ny//2-1)),
                 np.float32(0.), np.float32(dz*(Nz-1.)),
                 np.float32(1.*lam/n0),
                 np.float32(alpha1),
                 np.float32(alpha2),
                 kxs_g.data,
                 kys_g.data,
                 np.int32(len(kxs)),
                 np.float32(sigma)
                 )

    u = u_g.get()
    ex = ex_g.get()
    ey = ey_g.get()
    ez = ez_g.get()

    u_all = np.empty((Nz0, Ny, Nx), np.float32)
    ex_all = np.empty((Nz0, Ny, Nx), np.complex64)
    ey_all = np.empty((Nz0, Ny, Nx), np.complex64)
    ez_all = np.empty((Nz0, Ny, Nx), np.complex64)

    sz = [slice(0, Nz), slice(Nz, Nz0)]

    # spreading the calculated half plane to the full volume
    for i in [0, 1]:

        u_all[sz[1-i]] = u[1-i:Nz-1+i][::(-1)**i]
        if i==0:
            ex_all[sz[1-i]] = ex[1-i:Nz-1+i][::(-1)**i]
            ey_all[sz[1-i]] = ey[1-i:Nz-1+i][::(-1)**i]
            ez_all[sz[1-i]] = ez[1-i:Nz-1+i][::(-1)**i]
        else:
            ex_all[sz[1-i]] = np.conjugate(ex[1-i:Nz-1+i][::(-1)**i])
            ey_all[sz[1-i]] = np.conjugate(ey[1-i:Nz-1+i][::(-1)**i])
            ez_all[sz[1-i]] = np.conjugate(ez[1-i:Nz-1+i][::(-1)**i])

    print("time in secs:", time.time()-t)

    return u_all, ex_all, ey_all, ez_all


def focus_field_lattice_plane(shape=(256, 256),
                              units=(.1, .1),
                              z=0.,
                              lam=.5,
                              NA1=.4, NA2=.5,
                              sigma=.1,
                              kpoints=6,
                              n0=1.,
                              apodization_bound=10,
                              ex_g=None,
                              n_integration_steps=100):
    """calculates the complex 2d input field at position -z of a \
     for a bessel lattice beam.


    Parameters
    ----------

    shape: Nx,Ny
        the shape of the geometry
    units: dx,dy
        the pixel sizes in microns
    z:  float
        defocus position in microns, such that the beam would focus at z
        e.g. an input field with z = 10. would have its focal spot after 10 microns
    lam: float
        the wavelength of light used in microns
    NA1: float/list
        the numerical aperture of the inner ring
    NA2: float/list
        the numerical aperture of the outer ring
    sigma: float
        the standard deviation of the gaussian smear function applied to each point on the aperture
        (the bigger sigma, the tighter the sheet in y)
    kpoints: int/ (2,N) array
        defines the set of points on the aperture that create the lattice, can be
        - a (2,N) ndarray, such that kpoints[:,i] are the coordinates of the ith point
        - a single int, defining points on a regular polygon (e.g. 4 for a square lattice, 6 for a hex lattice)
        :math:`k_i = \\arcsin\\frac{NA_1+NA_2}{2 n_0} \\begin{pmatrix} \\cos \\phi_i \\\\ \\sin \\phi_i \\end{pmatrix}\quad, \\phi_i = \\frac{\\pi}{2}+\\frac{2i}{N}`
    n0: float
        the refractive index of the medium
    apodization_bound: int
        width of the region where the input field is tapered to zero (with a hamming window) on the +/- x borders
    n_integration_steps: int
        number of integration steps to perform
    return_all_fields: boolean
        if True, returns u,ex,ey,ez where ex/ey/ez are the complex vector field components

    Returns
    -------
    u: ndarray
        the 2d complex field
    Example
    -------

    >>> u = focus_field_lattice_plane((128,128), (0.1,0.1), z = 2., lam=.5, NA1 = .44, NA2 = .55, kpoints = 6)

    See also
    --------
    biobeam.focus_field_lattice: the corresponding 3d function

    """

    p = OCLProgram(absPath("kernels/psf_lattice.cl"),
                   build_options=["-I", absPath("kernels"), "-D", "INT_STEPS=%s"%n_integration_steps])

    Nx, Ny = shape
    dx, dy = units

    alpha1 = np.arcsin(1.*NA1/n0)
    alpha2 = np.arcsin(1.*NA2/n0)

    if np.isscalar(kpoints):
        kxs, kys = np.arcsin(.5*(NA1+NA2)/n0)*_poly_points(kpoints)
    else:
        kxs, kys = 1.*kpoints/n0

    if ex_g is None:
        use_buffer = False
        ex_g = OCLArray.empty((Ny, Nx), np.complex64)
    else:
        use_buffer = True

    assert ex_g.shape[::-1]==shape

    kxs_g = OCLArray.from_array(kxs.astype(np.float32))
    kys_g = OCLArray.from_array(kys.astype(np.float32))

    t = time.time()

    p.run_kernel("debye_wolf_lattice_plane", (Nx, Ny),
                 None,
                 ex_g.data,
                 np.float32(1.), np.float32(0.),
                 np.float32(-dx*(Nx-1)//2.), np.float32(dx*(Nx-1)//2.),
                 np.float32(-dy*(Ny-1)//2.), np.float32(dy*(Ny-1)//2.),
                 np.float32(-z),
                 np.float32(1.*lam/n0),
                 np.float32(alpha1),
                 np.float32(alpha2),
                 kxs_g.data,
                 kys_g.data,
                 np.int32(len(kxs)),
                 np.float32(sigma),
                 np.int32(apodization_bound),
                 )

    if not use_buffer:
        res = ex_g.get()
        print("time in secs:", time.time()-t)
        return res


if __name__=='__main__':
    u = focus_field_lattice((256,)*3,
                            (.1,)*3,
                            NA1=.44, NA2=.55,
                            sigma=.1,
                            kpoints=6,
                            lam=.488,
                            n0=1.33,
                            n_integration_steps=100)

    ex = focus_field_lattice_plane((512, 256), (.1,)*2,
                                   z=-12.8,
                                   # z = 0,
                                   NA1=.44, NA2=.55,
                                   sigma=.1,
                                   lam=.5,
                                   n0=1.33,
                                   apodization_bound=100,
                                   n_integration_steps=500)
