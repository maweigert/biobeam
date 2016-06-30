"""

this is the main module defining 3d psf functions and initial focus fields


mweigert@mpi-cbg.de

"""


from bpm.psf._focus_fields_debye import focus_field_debye, focus_field_debye_plane
from bpm.psf._focus_fields_cylindrical import focus_field_cylindrical, focus_field_cylindrical_plane
from bpm.psf._focus_fields_lattice import focus_field_lattice, focus_field_lattice_plane


import numpy as np

__all__ =["psf","psf_u0","psf_lightsheet","psf_cylindrical","psf_cylindrical_u0", "psf_lattice_u0"]

def psf(shape,units,lam, NA, n0 = 1.,
              n_integration_steps = 200,
              return_field = False):
    """

    Parameters
    ----------
    shape: Nx,Ny,Nz
        the shape of the geometry
    units: dx,dy,dz
        the pixel sizes in microns
    lam: float
        the wavelength
    NA
    n0
    n_integration_steps
    return_field

    Returns
    -------

    calculates the 3d psf for a perfect, aberration free optical system
    via the vectorial debye diffraction integral

    the psf is centered at a grid of given size with voxelsizes units



    see [1]_


    returns:
    u, the (not normalized) intensity

    or if return_field = True
    u,ex,ey,ez

    NA can be either a single number or an even length list of NAs (for bessel beams), e.g.
    NA = [.1,.2,.5,.6] lets light through the annulus .1<.2 and .5<.6

    References
    ----------

    .. [1] Matthew R. Foreman, Peter Toeroek, *Computational methods in vectorial imaging*, Journal of Modern Optics, 2011, 58, 5-6, 339

    """

    u, ex, ey, ez = focus_field_debye(shape = shape, units = units,
                                   lam = lam, NA = NA, n0 = n0,
                                   n_integration_steps = n_integration_steps)
    if return_field:
        return u,ex, ey, ez
    else:
        return u



def psf_lightsheet(shape,units,lam_illum,NA_illum, lam_detect, NA_detect, n0 = 1.,
              n_integration_steps = 200,
              return_field = False):
    """
    """

    u_detect= psf(shape = shape, units = units,
                         lam = lam_detect,
                         NA = NA_detect,
                         n0 = n0,
                        n_integration_steps= n_integration_steps)



    u_illum= psf(shape = shape[::-1],
                       units = units[::-1],
                         lam = lam_illum,
                         NA = NA_illum,
                         n0 = n0,
                        n_integration_steps= n_integration_steps)

    u_illum = u_illum.transpose((2,1,0))

    return u_detect*u_illum





def psf_u0(shape,units,zfoc = 0,NA = .4,lam = .5, n0 = 1., n_integration_steps = 200):
    """calculates initial plane u0 of a beam focused at zfoc
    shape = (Nx,Ny)
    units = (dx,dy)
    NAs = e.g. (0,.6)
    """

    Nx, Ny = shape
    dx, dy = units


    ex = focus_field_debye_plane(shape = (Nx,Ny),
                                           units = (dx,dy),
                                           z=-zfoc,
                              lam = lam,NA = NA,n0=n0,
                             n_integration_steps=n_integration_steps)

    return ex


def psf_cylindrical(shape,units,lam,NA, n0=1.,
                    return_field = False,
                    n_integration_steps = 100):
    """returns psf of cylindrical lerns with given NA
    """
    u, ex, ey, ez = focus_field_cylindrical(shape = shape, units = units,
                                   lam = lam, NA = NA, n0 = n0,
                                   n_integration_steps = n_integration_steps,
                                    )


    if return_field:
        return u,ex
    else:
        return u


def psf_cylindrical_u0(shape, units, zfoc = 0., lam=.5, NA=.4, n0=1.,  n_integration_steps = 200):
    """calculates initial plane u0 of a cylidrical lens beam with defocus zfoc
     (e.g. such that the focus is after propagation of zfoc )

    shape = (Nx,Ny)
    units = (dx,dy)
    NA = e.g. 0.6
    """

    Nx, Ny = shape
    dx, dy = units


    ex = focus_field_cylindrical_plane(shape = (Nx,Ny),
                                           units = (dx,dy),
                                           z=-zfoc,
                              lam = lam,NA = NA,n0=n0,
                             n_integration_steps=n_integration_steps)

    return ex



def psf_lattice_u0(shape, units, zfoc = 0.,
                   lam=0.5,
                   NA1= .4,
                   NA2 = .5,
                   sigma = .1,
                   apodization_bound = 0,
                   n0=1.,  n_integration_steps = 200):
    """bessel lattice defocus zfoc
     (e.g. such that the focus is after propagation of zfoc )

    shape = (Nx,Ny)
    units = (dx,dy)
    NA = e.g. 0.6
    """

    Nx, Ny = shape
    dx, dy = units


    ex = focus_field_lattice_plane(shape = (Nx,Ny),
                                    units = (dx,dy),
                                           z=-zfoc,
                              lam = lam,
                                   NA1 = NA1,
                                   NA2  =NA2,
                                   sigma = sigma,
                                   n0=n0,
                                   apodization_bound=apodization_bound,
                             n_integration_steps=n_integration_steps)

    return ex

if __name__ == '__main__':


    ex = psf_cylindrical_u0((128,)*2,(.1,)*2,zfoc = 6.4)

    ex = psf_lattice_u0((128,)*2,(.1,)*2,zfoc = 6.4, apodization_bound=20)