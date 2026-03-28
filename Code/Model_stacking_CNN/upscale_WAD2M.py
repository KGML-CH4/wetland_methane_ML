import numpy as np
import matplotlib.pyplot as plt
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random
import sys
import subprocess as sp
import gc
import rasterio
import netCDF4 as nc
from skimage.measure import block_reduce
import config
import utils


rep = int(sys.argv[1])

### params
start_year, end_year = 2006, 2018  # these are the years of fluxnet data
num_years=end_year-start_year+1
timesteps_per_year=12
timesteps=timesteps_per_year*num_years
num_windows=int(np.floor((timesteps-24)/12+1))   # ***DON'T TOUCH—model does not automatically scale to different window sizes***
days_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]
nonmissing_required = 4
shift_size = 6  # six month shift for southern hemisphere

### load global (+TEM) data
data0 = torch.load(config.fp_upscale_prep, weights_only=False)
X_sim = data0['X']
Y_sim = data0['Y']
Z_sim = data0['Z']
X_vars_sim = data0['X_vars']
Z_vars_sim = data0['Z_vars']
X_stats_sim = data0['X_stats']
Y_stats_sim = data0['Y_stats']
Z_coords = np.nanmax(Z_sim, axis=1)
longlat = np.nanmax(Z_sim, axis=1)
num_sites = len(X_sim)
del data0
del Z_sim
gc.collect()

### swap -9999 for nan
X_sim[X_sim == -9999] = np.nan
Y_sim[Y_sim == -9999] = np.nan

### undo original TEM normalization
Y_sim[:,:,0] = utils.Z_norm_reverse(Y_sim[:,:,0], Y_stats_sim[0,:])
for v in range(len(X_vars_sim)):
    X_sim[:,:,v] = utils.Z_norm_reverse(X_sim[:,:,v], X_stats_sim[v,:])

### Separate out "area" variable

# create new var
aind = list(X_vars_sim).index("area")
area_sim = X_sim[:,:,aind]
area_sim = np.nanmax(area_sim, axis=1)
area_vars_sim = X_vars_sim[aind]

# remove from X
X_sim = np.delete(X_sim, aind, axis=2)
X_vars_sim = np.delete(X_vars_sim, aind)

### add on the TEM estimate
X_sim = torch.concatenate([X_sim, Y_sim], dim=-1)

### normalize using observed mean and sd
data0 = torch.load(config.fp_prep_fluxnet, weights_only=False)
X_stats_obs = data0['X_stats']
X_vars = data0['X_vars']
Y_stats_obs = data0['Y_stats']
I_stats_obs = data0['I_stats']

# re-order
inds = [1, 2, 4, 5, 6, 7] 
X_stats_obs = X_stats_obs[inds,:]

for v in range(len(X_stats_obs)): 
    X_sim[:,:,v] = (X_sim[:,:,v] - X_stats_obs[v,0]) / X_stats_obs[v,1]

### convert nans back to -9999
X_sim = torch.nan_to_num(X_sim, nan=-9999)

### load new FRIN
def get_area(resolution):  # (google)
    R = 6_371_000  # average radius of Earth (meters) https://en.wikipedia.org/wiki/Earth_radius
    dlat = float(resolution)  # degrees lat/lon
    dlon = float(resolution)
    lat_edges = np.arange(-90, 90 + dlat, dlat)
    lon_edges = np.arange(-180, 180 + dlon, dlon)    
    lat_edges_rad = np.deg2rad(lat_edges)
    lon_edges_rad = np.deg2rad(lon_edges)
    dphi = np.diff(lat_edges_rad)
    dlambda = np.diff(lon_edges_rad)
    area = (R**2) * np.outer(np.diff(np.sin(lat_edges_rad)), dlambda)
    return area
    
def load_newFRIN(fp, target_year):
    data = np.full((720, 360, 12), np.nan)
    nc_file = nc.Dataset(fp)  

    frin = nc_file['Fw']  # (252, 360, 720)
    frin = np.array(frin)
    frin[frin == -9999] = np.nan

    # flip north-south: index zero is 89.9 (northernmost)
    frin = frin[:, ::-1, :]  

    # convert to 0.5-degree res
    area = get_area(0.25)  # (720, 1440)
    area = np.expand_dims(area, 0)
    area_inundated = frin * area
    kernel_size = (1, 2, 2)
    area_inundated = block_reduce(area_inundated, block_size=kernel_size, func=np.nanmean)  # mean—not sum.
    area_inundated *= 4  # scale up to 4 grid cells (2x2) worth. This effectively FILLS IN missing data.
    area = get_area(0.5)  # (360, 720)
    area = np.expand_dims(area, 0)
    frin = area_inundated / area 
    
    # slice out current year
    start_month = 12 * (target_year - 2000)   # despite what the file attributes says, I'm assuming this is months since 2020
    end_month = start_month + 12
    frin = frin[start_month:end_month,:,:]    
    
    # fill in array
    for month in range(12):
        datum = frin[month, :, :]
        datum = np.swapaxes(datum, 0, 1)  # (720, 360)
        data[:,:,month] = datum

    # convert nan to zero
    data = np.nan_to_num(data, nan=0)

    return data

frin_file = output_path + "/frin_processed.npy"
if os.path.exists(frin_file):    
    newFRIN = np.load(frin_file)
else:
    year_range = end_year-start_year+1
    newFRIN = []
    for year in range(2006,2018+1):
        newFRIN.append(load_newFRIN(config.fp_WAD2M_map, year))
    #
    newFRIN = np.concatenate(newFRIN, axis=-1)
    np.save(frin_file, newFRIN)

### offset southern hemisphere by 6 months
shifted = newFRIN[:, 0:180, shift_size:].copy()  # new var of shifted data
newFRIN[:, 0:180, :] = np.nan  # replace old data with nans
newFRIN[:, 0:180, 0:-shift_size] = shifted.copy()  # shove in shifted data

def yearly(timeseries):
    days_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    tot = 0
    for month in range(12):
        if not np.isnan(timeseries[month]):  # skip nans
            tot += timeseries[month] * days_per_month[month]
    return tot

@torch.no_grad()
def upscale(sitedata, area, frin, I_obs, lat):
    windowed_estimates = []
    windowed_frins = []
    for it in range(num_windows):
        analyze = False  # default
        window_range_in = range(timesteps_per_year*it,timesteps_per_year*it+(timesteps_per_year*2))  # window of (smaller) input
        x_piece = sitedata[window_range_in, :].to(device)
        f_piece = frin[window_range_in]#.to(device)
        i_piece = I_obs[window_range_in]#.to(device)
                
        ## check if the window is appropriate to analyze (e.g., enough non-missing data)
        # year 1
        x_piece_1 = x_piece[0:timesteps_per_year, :]

        # first check: are inputs consistently missing for each time step? I would it expect the model to get tripped up otherwise.
        # (rather, instead of checking, just assigning missing to all inputs where there is at least one missing)
        x_missing_1 = (x_piece_1 == -9999).any(dim=1)  # check which months have missing inputs for any variable
        for month in range(timesteps_per_year):
            if x_missing_1[month] == True: # if any is missing, replace all with missing
                x_piece_1[month, :] = -9999                        

        # (skipping filter on missing data in year-1, i.e., it can be all missing)
                
        # year 2
        x_piece_2 = x_piece[timesteps_per_year:(timesteps_per_year*2), :]
        
        # second check: do positions of missing x values == missing y value positions?
        # We cannot expect the model to output a good estimate for, say, January, if there are no inputs.
        # But the reciprocal: we should keep the inputs even if no output, because that's more information for the RNN to work with.
        x_missing_2 = (x_piece_2 == -9999).any(dim=1)
        for month in range(timesteps_per_year):
            if x_missing_2[month] == True:
                x_piece_2[month, :] = -9999  # if any X's are missing, set the rest to missing to avoid confusing the model
        # 
        x_missing_2 = (x_piece_2 == -9999).any(dim=1)  # re-counting after adding nans
        
        # third check: do we have enough months with non-missing data in year 2?
        if torch.sum(x_missing_2) <= (timesteps_per_year-nonmissing_required):
            analyze = True
        
        ## forward pass
        if analyze == True:
            hidden = model.init_hidden(1)  # bsz=1
            X_input = torch.unsqueeze(x_piece, dim=0).to(torch.float32)
            I_input = torch.unsqueeze(i_piece, dim=0).to(torch.float32)
            pred, _ = model(I_input, X_input, hidden)
            pred = pred[0,timesteps_per_year:(timesteps_per_year*2),0]  # chop off first year
            f_piece = f_piece[timesteps_per_year:(timesteps_per_year*2)]
            pred = pred * Y_stats_obs[0,1] + Y_stats_obs[0,0]  # unnormalize
            windowed_estimates.append(pred)  # torch.Size([12]) 
            windowed_frins.append(f_piece)  # (12,)
        else:  # fill with missing data to avoid irregular shaped data frames
            empty = np.empty((12))
            empty[:] = np.nan
            windowed_estimates.append(torch.tensor(empty))
            windowed_frins.append(empty)

    if len(windowed_estimates) == 0:
        print("all missing data for grid cell", lat, long)
        site_est = np.nan
    else:

        ## convert units
        windowed_estimates = torch.stack(windowed_estimates, dim=0)  # torch.Size([12 years, 12 months])  ******* CHANGED NOV. 4
        windowed_frins = np.stack(windowed_frins, axis=0)
        windowed_estimates *= 10**6  # square-m to square-km (mg / km^2 day)    
        windowed_estimates *= 10**-15  # mg to Tg (Tg / km^2 day)    
        windowed_estimates *= area  # multiply by area (Tg / day)
        windowed_estimates *= windowed_frins  # fraction indundated

        # shift southern hemisphere back
        if lat < 0:
            windowed_estimates = windowed_estimates.reshape(12*12)
            shifted = windowed_estimates[0:-shift_size].clone()  # new var of shifted data
            windowed_estimates[:] = np.nan  # replace old data with nans
            windowed_estimates[shift_size:] = shifted.clone()  # shove in shifted data
            # check: >>> a = np.arange(90).reshape((3,30)).astype(float); shifted = deepcopy(a[:, shift_size:]); a[:] = np.nan; a[:,0:-shift_size] = deepcopy(shifted); shifted = deepcopy(a[:,0:-shift_size]); a[:] = np.nan; a[:,shift_size:] = deepcopy(shifted)
            windowed_estimates = windowed_estimates.reshape(12,12)

    return(windowed_estimates)

### model params
n_a=8 #hidden state number
n_l=2 #layer of gru
dropout = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device, flush=True)

bands =        ['sur_refl_b01',
               'sur_refl_b02',
               'sur_refl_b03',
               'sur_refl_b04',
               'sur_refl_b05',
               'sur_refl_b06',
               'sur_refl_b07',
               ]

### match coords used to pull down MODIS images to the new coords
TEM_preprocess_path = "/scratch.global/chriscs/KGML/Out/Emissions/vUPSCALE/fluxnet_sim_vSGFRIN.sav"
data0 = torch.load(TEM_preprocess_path, weights_only=False)
M_sim = data0['Z']
M_vars_sim = data0['Z_vars']
M_coords = np.nanmax(M_sim, axis=1)
print(len(M_coords))

coords2_dict = {tuple(coord): i for i, coord in enumerate(M_coords.tolist())}  # key: coord; value: corresponding MODIS image index
match_map = {}
for i, coord in enumerate(Z_coords.tolist()):
    key = tuple(coord)
    if key in coords2_dict:
        modis_idx = coords2_dict[key] 
        match_map[i] = modis_idx  # key: TEM index; value: corresponding MODIS image idex

print(len(match_map))

### subset the sites for parallel computing
shuffled_indices = np.arange(num_sites)  # shuffle—consistently across reps using same random seed
rng = np.random.default_rng(seed=123)
rng.shuffle(shuffled_indices)

# subset table for current rep                                                              
max_concurrent_requests = 1000
n = int(np.ceil(num_sites / max_concurrent_requests))   # 50788/1000=50.788

### the loop: predict on each grid cell  
for i in range(run*n, (run*n)+n):
    if i < num_sites:  # not all i's exist using range(it*n, (it*n)+n)                                                                                                                                                              
        site = shuffled_indices[i]
        site_output_fp = config.fp_upscale_out + "/estimates_site_" + str(site) + ".npy"
        if os.path.exists(site_output_fp):    
            print(site_output_fp, "exists")
        else:
            long,lat = longlat[site]
            long_ind,lat_ind = coords2index(long, lat) 
            
            # load MODIS image
            if site in match_map:
                folder = config.fp_modis_global + "/site_" + str(match_map[site]) + "/"
                I = []
                for band in range(len(bands)):
                    filename = folder + "/" + bands[band] + ".tif"
                    with rasterio.open(filename) as src:
                        data = src.read()
                        I.append(data)
                #
                I = np.stack(I, axis=1)  # torch.Size([156, 7, 112, 112])

                # normalize
                I[I==-9999] = np.nan
                for v in range(I.shape[1]):
                    I[:,v,:,:] = (I[:,v,:,:] - I_stats_obs[v,0]) / I_stats_obs[v,1]
                I = np.nan_to_num(I, nan=-9999)
                I = torch.tensor(I)

                # shift MODIS data 6 months
                if lat < 0:
                    shifted = I[shift_size:, :,:,:].clone()  # new var of shifted data
                    I[:] = np.nan  # replace old data with nans
                    I[0:-shift_size, :,:,:] = shifted.clone()  # shove in shifted data                
                                
                estimates = []
                for rep in range(100):
                    print("rep", rep, "site", site)
                
                    # load up model
                    fp = config.config.fp_train + "/production_rep_" + str(rep) + ".sav"
                    checkpoint=torch.load(fp, map_location=torch.device('cpu'), weights_only=False)
                    model=model_stack(7,n_a,n_l,1,dropout)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(device)
                    model.eval()    
                        
                    # tile across the big image
                    tiles = []
                    tile_size = 10  # model expects 10x10 images
                    nrows = int(np.floor(I.shape[-2]/tile_size))  
                    ncols = int(np.floor(I.shape[-1]/tile_size))
                    for row in range(nrows):
                        for col in range(ncols):
                            y = row*tile_size
                            x = col*tile_size
                            tile = I[:,:,y:y+tile_size,x:x+tile_size]  # torch.Size([156, 7, 10, 10])
                            tile = upscale(X_sim[site], 
                                                       area_sim[site], 
                                                       newFRIN[long_ind,lat_ind,:], 
                                                       tile, 
                                                       lat)  # torch.Size([12, 12])
                            tiles.append(tile)
                
                    # average across tiles (we're still inside a single rep)
                    tiles = np.stack(tiles, axis=-1)  # (12, 12, 121)
                    tiles = np.nanmean(tiles, axis=-1)  # (12, 12)
                    estimates.append(tiles)
                
                # save
                estimates = np.stack(estimates, axis=-1)  # (12, 12, 100)
                np.save(site_output_fp, estimates)

