"""


mweigert@mpi-cbg.de

"""

import numpy as np
from biobeam import Bpm3d
from time import time
from itertools import product

def time_single(dshape, sub_fac = 1, free_prop = True, fast_math = True):

    dn  = None if free_prop else np.zeros(dshape)
    simul_xy = tuple([sub_fac*d for d in dshape[1:][::-1]])
    simul_z = sub_fac

    m = Bpm3d(dn = dn, shape = dshape, units = (.1,)*3,
              simul_xy = simul_xy,
              simul_z = simul_z,
              fftplan_kwargs={"fast_math":fast_math})


    t = time()
    u = m.propagate(return_shape="last")
    t = time()-t

    geom = simul_xy + (dshape[0]*sub_fac,)
    print "geometry %s / %s\n    free_prop = %s\tfast_math = %s\t\n    time = %.2f ms"\
          %(dshape[::-1],geom, free_prop, fast_math,1000.*t)




if __name__ == '__main__':

    free_props = [True, False]
    fast_maths = [True, False]
    sub_facs= [1,2]
    Ns = (256,512)


    for (N, sub_fac, free_prop, fast_math) in product(Ns,sub_facs, free_props, fast_maths):
        time_single((N,)*3, sub_fac=sub_fac, free_prop=free_prop, fast_math=fast_math)