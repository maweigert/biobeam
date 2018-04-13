"""


mweigert@mpi-cbg.de

"""

from biobeam import SimLSM_Cylinder
from spimagine import read3dTiff

    #dn = read3dTiff("/Users/mweigert/python/bpm_projects/forward_model/data/sample_elegans_512.tif")
    #dn = dn.transpose(0,2,1).copy()
    #signal = 1.*dn.real

    dn = read3dTiff("/Users/mweigert/python/bpm_projects/forward_model/data/sample2_512_dn.tif")[128:-128]
    signal = read3dTiff("/Users/mweigert/python/bpm_projects/forward_model/data/sample2_512_signal.tif")[128:-128]



    #some point sources
    max_sig = np.amax(np.abs(signal))
    np.random.seed(0)
    for _ in range(4000):
        k,j,i = np.random.randint(dn.shape[0]),np.random.randint(dn.shape[1]),np.random.randint(dn.shape[2])
        signal[k,j,i] = 100.*max_sig



    #signal[146,::10,:] = 40.*max_dn

    if not "m" in locals():
        m = SimLSM_Base(dn = dn,
                        signal = signal,
                        NA_illum= .1,
                        NA_detect=.7,
                        units = (.2,)*3,
                        #simul_xy_detect=(512,512),
                        #simul_xy_illum=(512,512),
                        n_volumes=2,
                         )

    u = m.propagate_illum(cz = -10.)

    #h = m.psf((10.,0,0))
    #hs = m.psf_grid_z(10,grid_dim = (16,16), zslice = 16)

    im = m.simulate_image_z(cz=-10, zslice=16,
                            psf_grid_dim=(16,16),
                            conv_sub_blocks=(8,8),
                            conv_pad_factor=3,
                            )

