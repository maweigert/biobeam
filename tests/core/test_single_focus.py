from __future__ import absolute_import
import numpy as np
import numpy.testing as npt


from biobeam import Bpm3d, focus_field_beam

def test_focus(shape = (256,)*3, units = (.1,)*3, NA = .3, n0 = 1.):
    """ propagates a focused wave freely to the center
    """

    Nx, Ny, Nz = shape

    dx, dy , dz = .1, .1, .1

    lam = .5

    _,u_debye,  _, _ = focus_field_beam(shape, units, n0= n0, lam=lam, NA=NA, return_all_fields=True)

    u0 = u_debye[0]


    m = Bpm3d(shape = shape, units = units, lam = lam, n0 = n0)
    u = m.propagate(u0)

    #u = m.propagate(m.u0_beam(NA = NA))

    return u, u_debye

if __name__ == '__main__':
    pass
    #
    # Nx = 128
    # Ny = 256
    # Nz = 256
    #
    # dx ,dy, dz = .2, .1, .1
    # NA = .4
    # n0 = 1.1
    #
    #
    # u_bpm, u_anal = test_focus((Nx,Ny,Nz),(dx,dy,dz),NA = NA, n0 = n0)
    #
    #
    # print "L2 difference = %.10f"%np.mean(np.abs(u_bpm-u_anal)**2)
    #
    #
    #
    # import pylab
    # import seaborn
    # col = seaborn.color_palette()
    #
    # pylab.figure(1)
    # pylab.clf()
    # pylab.subplot(1,3,1)
    # pylab.imshow(np.abs(u_anal[:,Ny/2,:]), cmap = "hot", aspect = dz/dx)
    # pylab.grid("off")
    # pylab.axis("off")
    # pylab.title("anal,  NA = %s, n0 = %.2f"%(NA,n0))
    #
    # pylab.subplot(1,3,2)
    # pylab.imshow(np.abs(u_bpm[:,Ny/2,:]), cmap = "hot", aspect = dz/dx)
    # pylab.grid("off")
    # pylab.axis("off")
    # pylab.title("bpm,  NA = %s, n0 = %.2f"%(NA,n0))
    # pylab.subplot(1,3,3)
    # pylab.imshow(np.abs(u_bpm[:,Ny/2,:]-u_anal[:,Ny/2,:]), cmap = "gray", aspect = dz/dx)
    # pylab.grid("off")
    # pylab.axis("off")
    # pylab.title("difference")
    #
    #
    # pylab.figure(2)
    # pylab.clf()
    #
    # pylab.subplot(2,1,1)
    # pylab.plot(np.real(u_anal[:,Ny/2,Nx/2]), c = col[1],  label="analy")
    # pylab.plot(np.real(u_bpm[:,Ny/2,Nx/2]), c = col[0], label="bpm")
    # pylab.legend()
    # pylab.title("real")
    # pylab.subplot(2,1,2)
    # pylab.plot(np.imag(u_anal[:,Ny/2,Nx/2]), c = col[1],  label="analy")
    # pylab.plot(np.imag(u_bpm[:,Ny/2,Nx/2]), c = col[0], label="bpm")
    # pylab.legend()
    # pylab.title("imag")
    #
    #
    #
    # pylab.show()
    # pylab.draw()
    #
