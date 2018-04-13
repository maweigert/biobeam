"""


mweigert@mpi-cbg.de

"""
from biobeam import SimLSM_Cylindrical


def test_with_dn():
    shape = (128,512,256)
    units = (.1,.23,.45)

    m = SimLSM_Cylindrical(dn = dn,
                        #signal = signal,
                        NA_illum= .1,
                        NA_detect=.7,
                        size = (150,300,150),
                               n0 = 1.33,
                        # simul_xy_detect=(512,1024),
                        # simul_xy_illum=(1024,512),
                        n_volumes=1,
                         )

if __name__ == '__main__':
    pass