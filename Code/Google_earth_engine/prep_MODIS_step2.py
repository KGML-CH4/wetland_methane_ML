import numpy as np
import sys
import torch
import os
import config

### load TEM grid cells                                                              
data0 = torch.load(config.fp_prep_TEM, weights_only=False)
Z_sim = data0['Z']
num_sites = len(Z_sim)

### read in MODIS data
sums = np.array([0.]*7)  # 7 bands
denom = 0
for site in range(num_sites):
    fp = fp_modis_intermediate + "/site_" + str(site) + "/sums.npy"
    sums += np.load(fp)
    denom += 10*10*100*12*13  # 10x10 tile, 100 tiles, 12 mo, 13 years
        
### global mean
means = sums / denom
fp = config.fp_modis_tiles + "/global_means.npy"
np.save(fp, means)

