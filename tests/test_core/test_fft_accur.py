from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from  gputools import OCLArray, fft, fft_plan
from six.moves import range


def get_np(N = 256, niter=100, sig = 1.):
    np.random.seed(0)
    a = np.random.normal(0,sig,(N,N)).astype(np.complex64)
    b = (1.*a.copy()).astype(np.complex64)

    rels = []
    for _ in range(niter):
        b = np.fft.ifftn(np.fft.fftn(b).astype(np.complex64)).astype(np.complex64)
        rels.append(np.amax(np.abs(a-b))/np.amax(np.abs(a)))

    return np.array(rels)

    
def get_gpu(N = 256, niter=100, sig = 1.):
    np.random.seed(0)
    a = np.random.normal(0,sig,(N,N)).astype(np.complex64)
    b = (1.*a.copy()).astype(np.complex64)

    c_g = OCLArray.empty_like(b)
    b_g = OCLArray.from_array(b)
    p = fft_plan((N,N), fast_math = False)
    
    rels = []
    for _ in range(niter):
        fft(b_g,res_g = c_g, plan = p)
        fft(c_g, res_g = b_g, inverse = True, plan = p)

        # b = fft(fft(b), inverse = True)
        # rels.append(np.amax(np.abs(a-b))/np.amax(np.abs(a)))
        rels.append(np.amax(np.abs(a-b_g.get()))/np.amax(np.abs(a)))

    return np.array(rels)


def test_parseval():
    Nx = 512
    Nz  = 100
    d = np.random.uniform(-1,1,(Nx,Nx)).astype(np.complex64)
    d_g = OCLArray.from_array(d.astype(np.complex64))

    s1, s2 = [],[]
    for i in range(Nz):
        print(i)
        fft(d_g, inplace=True, fast_math=False)
        fft(d_g, inverse = True,inplace=True,fast_math=False)
        s1.append(np.sum(np.abs(d_g.get())**2))

    for i in range(Nz):
        print(i)
        d = np.fft.fftn(d).astype(np.complex64)
        d = np.fft.ifftn(d).astype(np.complex64)
        s2.append(np.sum(np.abs(d)**2))

    return s1, s2

if __name__ == '__main__':

    s1, s2 = test_parseval()
    # N = 256
    #
    #
    #
    # r1 = get_np(N,20)
    # r2 = get_gpu(N,20)
    #
    #
    #
