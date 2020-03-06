import sys
import os

import scipy.ndimage as nd
import scipy.io as io
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import skimage.measure as sk
from mpl_toolkits import mplot3d

vox = np.ones((13, 6, 20))


def plotFromVoxels(voxels, file_path, threshold=0.5, reso=31):
    z, x, y = np.where(voxels >= threshold)
    fig = plt.figure()
    fig.set_size_inches(20.0 / 3, 20.0 / 3)
    ax = fig.gca(projection='3d')
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
    fig.savefig(file_path, format='png', transparent=True, dpi=300, pad_inches=0, bbox_inches='tight')
