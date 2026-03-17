import numpy as np
import sys
import torch
import os
import config

input_folder = config.wd + "/MODIS_tiles_TEM/Intermediate_step1_preprocessing/"
output_folder = config.wd + "/MODIS_tiles_TEM/Preprocessed_tiles/"

### load TEM grid cells                                                              
TEM_preprocess_path = config.wd + "/Out/prep_TEM.sav"
data0 = torch.load(TEM_preprocess_path, weights_only=False)
Z_sim = data0['Z']
num_sites = len(Z_sim)

### read in MODIS data
sums = np.array([0.]*7)  # 7 bands
denom = 0
for site in range(num_sites):
    fp = input_folder + "/site_" + str(site) + "/sums.npy"
    sums += np.load(fp)
    denom += 10*10*100*12*13  # 10x10 tile, 100 tiles, 12 mo, 13 years
        
### global mean
means = sums / denom
fp = output_folder + "/global_means.npy"
np.save(fp, means)

