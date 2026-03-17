import numpy as np
import sys
import torch
import os
import config

### load TEM grid cells                                                              
data0 = torch.load(config.fp_prep_TEM, weights_only=False)
Z_sim = data0['Z']
num_sites = len(Z_sim)

### read in data
ssds = np.array([0.]*7)  # 7 bands
denom = 0
for i in range(100):  # 100 tiles
    fp = config.fp_modis_tiles + "/ssds_" + str(i) + ".npy"
    ssds += np.load(fp)
    denom += 10*10*12*13*num_sites  # 10x10 tile, 12 mo, 13 years, 12519 sites
        
### global mean
SDs = np.sqrt(ssds / denom)
fp = config.fp_modis_tiles + "/global_SDs.npy"
np.save(fp, SDs)
