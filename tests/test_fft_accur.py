import numpy as np
from  gputools import OCLArray, fft, fft_plan


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

    

if __name__ == '__main__':

    
    N = 256


    
    r1 = get_np(N,20)
    r2 = get_gpu(N,20)


    
