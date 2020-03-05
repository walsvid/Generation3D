import sys
import os

import scipy.ndimage as nd
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import skimage.measure as sk
from mpl_toolkits import mplot3d

vox = np.ones((13, 6, 20))


def plotFromVoxels(voxels, reso=31):
    z, x, y = voxels.nonzero()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    major_ticks = np.arange(0, reso + 1, 4)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_zticks(major_ticks)
    ax.set_xlim(0, reso)
    ax.set_ylim(0, reso)
    ax.set_zlim(0, reso)
    ax.scatter(z, y, x, zdir='z', c='red', marker='s', alpha=0.5)
    plt.margins(0)
    plt.show()
