"""


mweigert@mpi-cbg.de

"""
from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from biobeam.simlsm.simlsm import SimLSM_Base
from six.moves import range


class SimLSM_DSLM(SimLSM_Base):

    def _prepare_u0_illum(self, zfoc):
        self.u0_illum = self._bpm_illum.u0_beam(NA = self.NA_illum, zfoc = zfoc)


    def propagate_illum_single(self,cz = 0, **bpm_kwargs):
        bpm_kwargs.update({"return_comp":"intens"})
        offset = int(cz/self._bpm_illum.dy)
        assert abs(offset)<= self.u0_illum.shape[0]//2

        print("offset: ",offset)
        u0 = np.roll(self.u0_illum, offset ,axis=0)
        u = self._bpm_illum.propagate(u0,**bpm_kwargs)
        return self._trans_illum(u, inv = True)

    def propagate_illum(self,cz = 0,  dx_parallel = None,**bpm_kwargs):
        bpm = self._bpm_illum
        bpm_kwargs.update({"return_comp":"intens"})
        offset = int(cz/bpm.dy)
        assert abs(offset)<= self.u0_illum.shape[0]//2
        print("offset",  offset)


        u0_base = np.roll(self.u0_illum, offset ,axis=0)


        #prepare the parallel scheme
        max_NA = self.NA_illum if np.isscalar(self.NA_illum) else max(self.NA_illum)

        if dx_parallel is None:
            dx_parallel = 2*bpm.lam/max_NA

        print("dslm prop with parallelize beams of distance dx = %s mu"%dx_parallel)

        # the beamlet centers in the simul_xy coordinates
        ind_step = int(np.ceil(1.*dx_parallel/bpm.dx))
        #make sure its divisible by the grid size dimension
        ind_step = [i for i in range(ind_step, bpm.simul_xy[0]+1) if bpm.simul_xy[0]%i==0][0]

        if ind_step<=0:
            raise ValueError("dx resolution too coarse to propagate in parallel pleae increase dx_parallel")


        inds = np.arange(0,bpm.simul_xy[0],ind_step)

        print(inds,ind_step, bpm.simul_xy)


        u0 = np.sum([np.roll(u0_base,i,axis=1) for i in inds],axis=0)



        u = None
        # now scan
        for i in range(ind_step):
            print("propagating beamlets %s/%s"%(i+1,ind_step))
            u_part = bpm.propagate(u0,**bpm_kwargs)
            u = u_part if u is None else u+u_part


            u0 = np.roll(u0,1,axis=1)

        return self._trans_illum(u, inv = True)




if __name__ == '__main__':

    dn = np.zeros((256,512,256))

    signal = np.zeros_like(dn)

    #some point sources
    np.random.seed(0)
    for _ in range(4000):
        k,j,i = np.random.randint(dn.shape[0]),np.random.randint(dn.shape[1]),np.random.randint(dn.shape[2])
        signal[k,j,i] = 1.


    if not "m" in locals():
        m = SimLSM_DSLM(dn = dn,
                        signal = signal,
                        NA_illum= .4,
                        NA_detect=.7,
                        units = (.4,)*3,
                        #simul_xy_detect=(512,512),
                        #simul_xy_illum=(512,1024),
                         )

    u1 = m.propagate_illum(cz = -10)
    u2 = m.propagate_illum(cz = 10)

    h = m.psf((0.,0,0))

    #
    # im = m.simulate_image_z(cz=0, zslice=16,
    #                         psf_grid_dim=(16,16),
    #                         conv_sub_blocks=(8,8),
    #                         conv_pad_factor=3,
    #                         )
    #
    #
