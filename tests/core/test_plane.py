"""the main method for beam propagation in media with coated spheres"""

import numpy as np
import numpy.testing as npt


from biobeam import Bpm3d

def cmp_plane(shape = (256,)*3, n_x_comp = 0, n0 = 1., n = None):
    """ propagates a plane wave freely
    n_x_comp is the tilt in x
    """
    Nx, Ny, Nz = shape

    dx, dz = .05, 0.05

    if n is None:
        n = n0
        
    lam = .5



    #enforce periodicity...
    units = (dx,dx,dz)

    x = dx*np.arange(Nx)
    y = dx*np.arange(Ny)
    z = dz*np.arange(Nz)
    Z,Y,X = np.meshgrid(z,y,x,indexing="ij")

    
    k_x = 1.*n_x_comp/(dx*(Nx-1.))

    k_y = 0.

    
    k_z = np.sqrt(1.*n**2/lam**2-k_x**2-k_y**2)


    
    u_plane = np.exp(2.j*np.pi*(k_z*Z+k_y*Y+k_x*X))

    dn = (n-n0)*np.ones_like(Z)

    m = Bpm3d(shape = shape, units = units, dn = dn, n0 = n0, lam = lam)
    u = m._propagate_single(u0 = u_plane[0])

    l2_diff  = np.mean(np.abs(u_plane-u)**2)
    print "shape =%s x_comp: %s n0=%.2f \t\tL2 difference = %.4f"%(shape,n_x_comp, n0, l2_diff)

    npt.assert_almost_equal(np.mean(np.abs(u_plane-u)**2),0,decimal = 2)

    return u, u_plane


def plot_some():

    Nx = 256
    Ny = 128
    Nz = 128

    x_comps = [0,2]
    n0s = [1.,1.2]

    u_bpm, u_plane = [],[]
    for x_comp,n0 in zip(x_comps, n0s):
        u1,u2 = cmp_plane((Nx,Ny,Nz),n_x_comp=x_comp, n0 = n0)
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



    pylab.legend()
    pylab.show()
    pylab.draw()



def test_some():
    from itertools import product

    Nx = 256
    Ny = 128
    Nz = 128

    x_comps = [0,1,4]
    n0s = [1.,1.2]
    shapes = [(128,128,100),(256,128,60),(256,128,300)]


    for x_comp,n0,shape in product(x_comps, n0s, shapes):
        print x_comp, n0, shape
        cmp_plane(shape,n_x_comp=x_comp, n0 = n0)
        # u_bpm.append(u1)
        # u_plane.append(u2)




if __name__ == '__main__':

    test_some()


    #plot_some()

