import numpy as np
import sys
import torch
import os
import config

tile_index = sys.argv[1]

### read mean
fp = config.fp_modis_tiles + "/global_means.npy"
means = np.load(fp)

### load TEM grid cells                                                              
data0 = torch.load(config.fp_prep_TEM, weights_only=False)
Z_sim = data0['Z']
num_sites = len(Z_sim)

### read tiles
ssds = np.array([0.]*7)  # 7 bands
col = []
for site in range(num_sites):
    fp = config.fp_modis_intermediate + "/site_" + str(site) + "/tile_" + str(tile_index) + ".npy"
    tile = np.load(fp)  # (156, 7, 10, 10)
    col.append(tile)
    for band in range(7):
        ssd = np.sum( (tile[:, band, :, :]-means[band])**2 )
        ssds[band] += ssd

# write
col = np.array(col)
fp = config.fp_modis_tiles + "/tile_" + str(tile_index) + ".npy"
np.save(fp, col)     

### save ssds
fp = config.fp_modis_tiles + "/ssds_" + str(tile_index) + ".npy"
np.save(fp, ssds)
