import numpy as np
import sys
import torch
import os
import config

tile_index = sys.argv[1]

input_folder = config.wd + "/Out/MODIS_tiles_TEM/Intermediate_step1_preprocessing/"
output_folder = confid.wd + "/Out/MODIS_tiles_TEM/Preprocessed_tiles/"
os.makedirs(output_folder, exist_ok=True)

### read mean
fp = output_folder + "/global_means.npy"
means = np.load(fp)

### load TEM grid cells                                                              
TEM_preprocess_path = config.wd + "/Out/prep_TEM.sav"
data0 = torch.load(TEM_preprocess_path, weights_only=False)
Z_sim = data0['Z']
num_sites = len(Z_sim)

### read tiles
ssds = np.array([0.]*7)  # 7 bands
col = []
for site in range(num_sites):
    fp = input_folder + "/site_" + str(site) + "/tile_" + str(tile_index) + ".npy"
    tile = np.load(fp)  # (156, 7, 10, 10)
    col.append(tile)
    for band in range(7):
        ssd = np.sum( (tile[:, band, :, :]-means[band])**2 )
        ssds[band] += ssd

# write
col = np.array(col)
fp = output_folder + "/tile_" + str(tile_index) + ".npy"
np.save(fp, col)     

### save ssds
fp = output_folder + "/ssds_" + str(tile_index) + ".npy"
np.save(fp, ssds)
