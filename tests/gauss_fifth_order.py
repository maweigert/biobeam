"""


mweigert@mpi-cbg.de

"""


from numpy import *


if __name__ == '__main__':

    NA = .3
    dx = .1
    Nx, Nz = 256, 256
    x = dx*(arange(Nx)-Nx/2)


    z = dx*(arange(Nz)-Nz/2)

    Y,X = meshgrid(x,x,indexing = "ij")

    R = hypot(X,Y)

    lam = .5
    k = 2*pi/lam

    #beam parameter w0
    w0 = 2./k/arcsin(NA)


    s = 1./k/w0

    #normalized coords
    X2 = 1.*X/w0
    Y2 = 1.*Y/w0

    R2 = hypot(X2,Y2)

    def E_plane(z):
        z2 = 1.*z/k/w0**2
        Q = 1./(1.j+2*z2)
        phi0 = 1.j*Q*exp(-1.j*R2**2*Q)

        Ex = (1+s**2*(-R2**2*Q**2+1.j*R2**4*Q**3-2.*Q**2*X2**2)+\
          s**4*(2*R2**4*Q**4-3.j*R2**6*Q**5-.5*R2**8*Q**6+(8*R2**2*Q**4-2.j*R2**4*Q**5)*X2**2))\
        *phi0*exp(-1.j*z/s**2)

        Ey = (-2.*s**2*Q**2*X2*Y2+s**4*X2*Y2*(8*R2**2*Q**4-2.j*R2**4*Q**5))\
        *phi0*exp(-1.j*z/s**2)

        Ez = (-2.*s*Q*X2+s**3*X2*(6.*s**2*Q**3-2.j*R2**4*Q**4)+\
          s**5*X2*(-20.*R2**4*Q**5+10.j*R2**6*Q**6+R2**8*Q**7))\
        *phi0*exp(-1.j*z/s**2)

        return Ex,Ey,Ez

    Ex,Ey,Ez = E_plane(0)

    field = stack([stack(E_plane(_z)) for _z in z]).transpose((1,0,2,3))