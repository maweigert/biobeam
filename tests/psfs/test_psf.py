"""


mweigert@mpi-cbg.de
"""

from __future__ import print_function, unicode_literals, absolute_import, division

import os
import re
import numpy as np
from biobeam import focus_field_beam
from tifffile import imread
import matplotlib.pyplot as plt


def compare_plot(fname_base):
    try:
        NA = float(re.findall("NA_(.*?)_", os.path.basename(fname_base))[0])
        print("NA = ", NA)
        dx = float(re.findall("dx_(.*?)_", os.path.basename(fname_base))[0])
        print("dx = ", dx)
        n0 = float(re.findall("n0_(.*?)_", os.path.basename(fname_base))[0])
        print("n0 = ", n0)
    except Exception as e:
        print(e)
        print("could not parse %s " % fname_base)

    gt_xy = imread(fname_base + "xy.tif")[:-1, :-1]
    gt_xy *= 1. / np.amax(gt_xy)

    gt_xz = imread(fname_base + "xz.tif")[:-1, :-1]
    gt_xz *= 1. / np.amax(gt_xz)

    Ny, Nx = gt_xy.shape
    Nz, _ = gt_xz.shape

    x = dx * (np.arange(Nx) - Nx // 2)
    z = dx * (np.arange(Nz) - Nz // 2)

    u = focus_field_beam((Nx, Ny, Nz), (dx,) * 3, NA=NA, n0=n0)
    u *= 1. / np.amax(u)

    # psf width
    _p = gt_xy[Ny // 2].copy()
    _p *= 1. / np.sum(_p)/dx
    sigma_x = 2.*np.sqrt(dx*np.sum(x ** 2 * _p) - dx*np.sum(x * _p) ** 2)
    _p = gt_xz[:, Nz // 2].copy()
    _p *= 1. / np.sum(_p)/dx
    sigma_z = 2.*np.sqrt(dx*np.sum(x ** 2 * _p) - dx*np.sum(x * _p) ** 2)
    print(sigma_x, sigma_z)

    plt.cla()
    plt.plot(x, gt_xy[Ny // 2], color="C0", label="x - psflab")
    plt.plot(x, u[Nz // 2, Ny // 2], ".:", color="C0", label="x - biobeam")

    plt.plot(z, gt_xz[:, Nx // 2], color="C1", ls=":", label="z - psflab")
    plt.plot(z, u[:, Ny // 2, Nx // 2], ".:", color="C1", label="z - biobeam")

    plt.title("NA = %s, n0 = %s\nsig_x = %.2f , sig_z = %.2f " % (NA, n0, sigma_x, sigma_z))

    plt.legend(prop={'size':8})


    return gt_xy, gt_xz, u


if __name__ == '__main__':
    fnames = ["data/psflab_NA_0.4_dx_0.1_n0_1.5_","data/psflab_NA_0.7_dx_0.1_n0_1.0_",
              "data/psflab_NA_1.1_dx_0.1_n0_1.33_", "data/psflab_NA_1.1_dx_0.1_n0_1.5_"]

    plt.figure(figsize = (5*len(fnames),4))

    for i,fname in enumerate(fnames):
        plt.subplot(1,len(fnames),i+1)
        gt_xy, gt_xz, u = compare_plot(fname)

    plt.show()