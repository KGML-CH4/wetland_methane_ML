import numpy as np
import sys
import torch
import os
import config

output_folder = config.wd + "/Out/MODIS_tiles_TEM/Preprocessed_tiles/"

### load TEM grid cells                                                              
TEM_preprocess_path = config.wd + "/Out/prep_TEM.sav"
data0 = torch.load(TEM_preprocess_path, weights_only=False)
Z_sim = data0['Z']
num_sites = len(Z_sim)

### read in data
ssds = np.array([0.]*7)  # 7 bands
denom = 0
for i in range(100):  # 100 tiles
    fp = output_folder + "/ssds_" + str(i) + ".npy"
    ssds += np.load(fp)
    denom += 10*10*12*13*num_sites  # 10x10 tile, 12 mo, 13 years, 12519 sites
        
### global mean
SDs = np.sqrt(ssds / denom)
fp = output_folder + "/global_SDs.npy"
np.save(fp, SDs)
