"""the main method for beam propagation in media with coated spheres"""

import numpy as np
import numpy.testing as npt


from bpm import bpm_3d

from bpm import bpm_3d

def test_plane(size = (256,)*3, n_x_comp = 0, n0 = 1., n = None):
    """ propagates a plane wave freely
    n_x_comp is the tilt in x
    """
    Nx, Ny, Nz = size

    dx, dz = .05, 0.05

    if n is None:
        n = n0
        
    lam = .5

    units = (dx,dx,dz)
    
    x = dx*np.arange(Nx)
    y = dx*np.arange(Ny)
    z = dz*np.arange(Nz)
    Z,Y,X = np.meshgrid(z,y,x,indexing="ij")

    
    k_x = 1.*n_x_comp/(dx*(Nx-1.))

    k_y = k_x

    
    k_z = np.sqrt(1.*n**2/lam**2-k_x**2-k_y**2)

    print (k_x,k_z), np.sqrt(k_x**2+k_y**2+k_z**2)
    
    u_plane = np.exp(-2.j*np.pi*(k_z*Z+k_y*Y+k_x*X))

    u = 0
    dn = (n-n0)*np.ones_like(Z)


    u = bpm_3d((Nx,Ny,Nz),units= units, lam = lam,
                   n0 = n0,
                   dn = dn,
                   n_volumes = 1,
                   absorbing_width=0,
                   u0 = u_plane[0,...])

    print "L2 difference = %.4f"%np.mean(np.abs(u_plane-u)**2)
    #npt.assert_almost_equal(np.mean(np.abs(u_plane-u)**2),0,decimal = 2)
    return u, u_plane

if __name__ == '__main__':



    Nx = 256
    Ny = 128
    Nz = 128

    x_comps = [0,2]
    n0s = [1.,1.2]

    u_bpm, u_plane = [],[]
    for x_comp,n0 in zip(x_comps, n0s):
        u1,u2 = test_plane((Nx,Ny,Nz),n_x_comp=x_comp, n0 = n0)
        u_bpm.append(u1)
        u_plane.append(u2)


    import pylab
    import seaborn
    pylab.ioff()
    col = seaborn.color_palette()

    n = len(u_bpm)


    pylab.figure(1)
    pylab.clf()
    for i in range(n):
        pylab.subplot(n,1,i+1)
        pylab.plot(np.real(u_plane[i][:,Ny/2,Nx/2]), "-",c = col[1],  label="analy")
        pylab.plot(np.real(u_bpm[i][:,Ny/2,Nx/2]), ".:", c = col[0], label="bpm")

        pylab.legend()
        pylab.title("x_comp = %s, n0 = %.2f"%(x_comps[i],n0s[i]))

    pylab.figure(2)
    pylab.clf()
    for i in range(n):
        pylab.subplot(n,2,2*i+1)
        pylab.imshow(np.real(u_plane[i][:,Ny/2,:]), cmap = "hot")
        pylab.grid("off")
        pylab.title("anal,  x_comp = %s, n0 = %.2f"%(x_comps[i],n0s[i]))

        pylab.subplot(n,2,2*i+2)
        pylab.imshow(np.real(u_bpm[i][:,Ny/2,:]), cmap = "hot")
        pylab.grid("off")
        pylab.title("bpm,  x_comp = %s, n0 = %.2f"%(x_comps[i],n0s[i]))

    # pylab.subplot(2,1,2)
    # pylab.title("x_comp = 2")
    # pylab.plot(np.real(a1)[:,64,64], "-", c = col[1], label="2 bpm")
    # pylab.plot(np.real(a2)[:,64,64], ".",c = col[1],  label="2 analy")


    pylab.legend()
    pylab.show()
    pylab.draw()

    # test_plane(1)
    # test_plane(2)
    
