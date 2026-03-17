#!/usr/bin/env python
# coding: utf-8

# In[75]:


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


# In[76]:


def Z_norm(X):
    X_mean=np.nanmean(X)
    X_std=np.nanstd(np.array(X))
    return (X-X_mean)/X_std, X_mean, X_std
    
def Z_norm_reverse(X,Xnorm,units_convert=1.0):
    return (X*Xnorm[1]+Xnorm[0])*units_convert

def coords2index(long, lat):
    # long coords range (-180, 180); lat coords range (-90,90) verified bashing the data files
    long_ind = long + 180  # (0,360)
    lat_ind = lat + 90  # (0,180)

    # both lat and long have 1/2 degree increments = 720 indices for each.    
    long_ind *= 2  # (0,720)
    lat_ind *= 2  # (0,360)

    # integer indices
    long_ind = int(long_ind)
    lat_ind = int(lat_ind)

    return long_ind, lat_ind


# In[77]:


### nn architecture

num_classes = 3

class cnn_branch(nn.Module):
    def __init__(self):
        super(cnn_branch, self).__init__()
        self.num_classes=num_classes
        self.conv1 = nn.Conv2d(in_channels=7, out_channels=16, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3))
        self.dense_1 = nn.Linear(128, 64)
        self.dense_2 = nn.Linear(64, self.num_classes)
        # self.dense_1 = nn.Linear(128, 1)
        # self.dense_2 = nn.Linear(timesteps_per_year*2, self.num_classes)
        self.pool = nn.MaxPool2d((2,2))
        
    def forward(self, images):
        B, T, C, H, W = images.shape  # [10, 24, 7, 10, 10]
        output = images.view(B * T, C, H, W)  # [240, 7, 10, 10]
        output = self.conv1(output)  # [240, 8, 8, 8]
        output = F.relu(output) 
        output = self.pool(output)  # [240, 16, 4, 4]
        output = self.conv2(output)  # [240, 16, 2, 2]
        output = F.relu(output) 

        # flattens all channels of each image for each training example x timestep
        output = output.flatten(start_dim=1)  # [240, 64]  
        output = self.dense_1(output)  # [240, 16]
        output = F.relu(output)
        output = self.dense_2(output)  # [240, 16]

        output = F.gumbel_softmax(output, tau=1.0, hard=True, dim=1)  # torch.Size([240, 4])
        output = output.view(B, T, -1)  # [10, 24, 4]
        
        return output


class pureML_GRU(nn.Module):
    def __init__(self, ninp, nhid, nlayers, nout1, dropout):
        super(pureML_GRU, self).__init__()
        self.num_classes=num_classes
        self.gru = nn.GRU(ninp+self.num_classes, nhid, nlayers,dropout=dropout, batch_first=True)
        self.densor_flux = nn.Linear(nhid, nout1)
        self.nhid = nhid
        self.nlayers = nlayers
        self.drop=nn.Dropout(dropout)
        self.init_weights()
        self.cnn_branch = cnn_branch()

    def init_weights(self):
        initrange = 0.1 #may change to a small value
        self.densor_flux.bias.data.zero_()
        self.densor_flux.weight.data.uniform_(-initrange, initrange)

    def forward(self, images, inputs, hidden):
        output = self.cnn_branch(images)  # torch.Size([10, 24, 4])
        output = torch.cat([inputs, output], dim=-1)  # torch.Size([10, 24, 6])
        output, hidden = self.gru(output, hidden) 
        output = self.drop(output)
        output = self.densor_flux(output)  # torch.Size([10, 24, 1])
        return output, hidden
        
    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.nlayers, bsz, self.nhid)


# In[78]:


### file paths
model_version = "UPSCALE"
pretrain_version = "161"

input_path = '/scratch.global/chriscs/KGML/Data/Emissions/'
output_path = '/scratch.global/chriscs/KGML/Out/Emissions/v' + model_version + "/"
stats_path = '/scratch.global/chriscs/KGML/Out/Emissions/v' + pretrain_version + "/fluxnet_obs_v" + pretrain_version + '.sav'
final_model_path = "/scratch.global/chriscs/KGML/Out/Emissions/v" + pretrain_version + "/"
modis_path = '/scratch.global/chriscs/KGML/Data/Emissions/MODIS_full_upscale/'


# input_path = '/Users/chris/TempWorkSpace/KGML/Data/Emissions/'
# output_path = '/Users/chris/TempWorkSpace/KGML/Out/Emissions/v' + model_version + "/"

path_save_obs = output_path + 'fluxnet_obs_v' + model_version + '.sav'
path_save_sim = output_path + 'fluxnet_sim_v' + model_version + '.sav'
pretrain_path = output_path + 'train_' + str(model_version) + '.sav'
finetune_path = output_path + 'finetune_' + str(model_version) + '.sav'
#path_shuffle = output_path + 'shuffle_inds_v' + model_version + '.sav'


# WAD2M
gridded_preprocess_path = '/scratch.global/chriscs/KGML/Out/Emissions/vUPSCALE/fluxnet_sim_vWAD2M.sav'
ml_output_path = output_path + "/WAD2M/"

# # SGFRIN
# gridded_preprocess_path = '/scratch.global/chriscs/KGML/Out/Emissions/vUPSCALE/fluxnet_sim_vSGFRIN.sav'
# ml_output_path = output_path + "/SGFRIN/"


# In[79]:


### params
start_year, end_year = 2006, 2018  # these are the years of fluxnet data
num_years=end_year-start_year+1
timesteps_per_year=12
timesteps=timesteps_per_year*num_years
num_windows=int(np.floor((timesteps-24)/12+1))   # ***DON'T TOUCH—model does not automatically scale to different window sizes***
days_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]
nonmissing_required = 4
shift_size = 6  # six month shift for southern hemisphere
print(num_windows)


# In[ ]:





# In[ ]:





# In[ ]:





# In[86]:


### load global (+TEM) data
data0 = torch.load(gridded_preprocess_path, weights_only=False)
# fp = "/scratch.global/chriscs/KGML/Out/Emissions/vUPSCALE/fluxnet_sim_vUPSCALE.sav_OG"
# data0 = torch.load(fp, weights_only=False)
X_sim = data0['X']
Y_sim = data0['Y']
Z_sim = data0['Z']
print(X_sim.shape, flush=True)
print(Y_sim.shape, flush=True)
print(Z_sim.shape, flush=True)
X_vars_sim = data0['X_vars']
Z_vars_sim = data0['Z_vars']
print(X_vars_sim, flush=True)
print(Z_vars_sim, flush=True)
X_stats_sim = data0['X_stats']
Y_stats_sim = data0['Y_stats']
Z_coords = np.nanmax(Z_sim, axis=1)
longlat = np.nanmax(Z_sim, axis=1)
print(Z_coords)
num_sites = len(X_sim)
print(num_sites)

del data0
del Z_sim
gc.collect()


# In[87]:


a = (X_sim == -9999)
a = torch.sum(a, axis=(0,2))
plt.plot(a)
plt.show()


# In[81]:


### swap -9999 for nan
X_sim[X_sim == -9999] = np.nan
Y_sim[Y_sim == -9999] = np.nan


# In[60]:


### undo original TEM normalization
Y_sim[:,:,0] = Z_norm_reverse(Y_sim[:,:,0], Y_stats_sim[0,:])
for v in range(len(X_vars_sim)):
    X_sim[:,:,v] = Z_norm_reverse(X_sim[:,:,v], X_stats_sim[v,:])


# In[61]:


### Separate out "area" variable

# create new var
aind = list(X_vars_sim).index("area")
area_sim = X_sim[:,:,aind]
area_sim = np.nanmax(area_sim, axis=1)
area_vars_sim = X_vars_sim[aind]
print(area_vars_sim, area_sim.shape)

# remove from X
X_sim = np.delete(X_sim, aind, axis=2)
X_vars_sim = np.delete(X_vars_sim, aind)

print(X_vars_sim)
print(X_vars_sim.shape)
print(X_sim.shape)


# In[62]:


### add on the TEM estimate
print(X_sim.shape)
print(Y_sim.shape)
X_sim = torch.concatenate([X_sim, Y_sim], dim=-1)
print(X_sim.shape)


# In[63]:


### normalize using observed mean and sd
data0 = torch.load(stats_path, weights_only=False)
X_stats_obs = data0['X_stats']
X_vars = data0['X_vars']
Y_stats_obs = data0['Y_stats']
I_stats_obs = data0['I_stats']
print(X_stats_obs.shape)

#  sim vars
# ['LE' 'temp' 'site_class_1' 'site_class_2' 'site_class_3' 'site_class_4']
#    0     1       2            3                 4                5
#
#  obs vars
#  ['FCH4' 'LE_F' 'TA_F' 'FCH4_F_ANNOPTLM' 'site_class_1' 'site_class_2' 'site_class_3' 'site_class_4' 'tem_flux']
#     0      1       2          3                 4             5              6             7             8


# re-order
inds = [1, 2, 4, 5, 6, 7] 
X_stats_obs = X_stats_obs[inds,:]
print(X_stats_obs.shape)

for v in range(len(X_stats_obs)): 
    X_sim[:,:,v] = (X_sim[:,:,v] - X_stats_obs[v,0]) / X_stats_obs[v,1]

print(X_sim.shape)


# In[64]:


### convert nans back to -9999

print(torch.sum(X_sim==-9999))
print(torch.sum(np.isnan(X_sim)))
X_sim = torch.nan_to_num(X_sim, nan=-9999)
print(torch.sum(X_sim==-9999))
print(torch.sum(np.isnan(X_sim)))


# In[65]:


# ### sg_FRIN
# def load_predictor(fpath, header, target_year, ch4=False):
#     print("loading predictor", fpath)
#     counter = 0
#     var = np.full((720, 360, 12), np.nan)
#     long_ind = header.index("LONG")  # index for "LONG" in the list of variables
#     lat_ind = header.index("LAT")
#     year_ind = header.index("YEAR")
#     with open(fpath) as infile:
#         for line in infile:
#             counter += 1
#             newline = line.strip().split(",")
#             year = int(newline[year_ind])                  
#             if year == target_year:
#                 long = float(newline[long_ind])  # the longitude coordinate of the grid cell
#                 lat = float(newline[lat_ind])
#                 long_cell, lat_cell = coords2index(long, lat)  # convert the lat long to array indices starting from 0 (instead of negatives)
#                 data = newline[len(header):len(newline)-1]  
#                 data = np.array(data).astype(np.float32)
#                 data[data == -99999.0] = 0
#                 data /= 10000
#                 var[long_cell, lat_cell, :] = data
#     #
#     return var




# fpath = input_path + "sg_frin.tem"    
# header = ["LONG", "LAT", "VAR_NAME", "DONTKNOW", "YEAR", "SUM", "min", "mean", "max"]
# year_range = end_year-start_year+1
# newFRIN = []
# for year in range(2006, 2018+1):
#     print(year)
#     if year <= 2012:
#         newFRIN.append(load_predictor(fpath, header, year))
#     else:  # use 2012 data for later years
#         newFRIN.append(load_predictor(fpath, header, 2012))
# #
# newFRIN = np.concatenate(newFRIN, axis=-1)
# print(np.nanmean(newFRIN))
# print(newFRIN.shape)


# # # plot for sanity check
# # import cartopy.crs as ccrs
# # import cartopy.feature as cfeature
# # import matplotlib.colors as colors
# # from matplotlib.colors import SymLogNorm
# # data = newFRIN
# # data = np.squeeze(data)
# # data[data ==-9999] = np.nan
# # data = np.nanmean(data, axis=-1)   # ************ correct index?***************
# # plt.hist(data.flatten(), bins=100)
# # plt.show()
# # data = data.T
# # lon = np.linspace(-180, 180, 720)
# # lat = np.linspace(-90, 90, 360)
# # fig = plt.figure(figsize=(11, 6))
# # ax = plt.axes(projection=ccrs.PlateCarree())    
# # heatmap = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
# #                         cmap='coolwarm')
# # ax.coastlines()  # Outline of continents
# # ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', alpha=0.3)
# # ax.add_feature(cfeature.OCEAN, facecolor='white', alpha=0.2)
# # ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
# # cbar = fig.colorbar(heatmap, ax=ax, orientation='horizontal',
# #                     pad=0.05,  # distance from plot
# #                     fraction=0.035,  # relative width of colorbar
# #                     shrink=0.7,  # shrink height (for horizontal, this affects thickness)
# #                     aspect=25)  # ratio of long to short axis
# # plt.tight_layout()
# # plt.show()


# In[66]:


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
        # datum = np.expand_dims(datum, axis=-1)  # (720, 360, 1)
        # datum = np.repeat(datum, 31, axis=-1)  # (720, 360, 31)
        # start_day = month*31
        # end_day = start_day + 31
        # data[:,:,start_day:end_day] = datum
        data[:,:,month] = datum

    # convert nan to zero
    #     (why?)
    data = np.nan_to_num(data, nan=0)  # I think this is currently missing in the sgfrin version
    
    
    #    
    return data

frin_file = output_path + "/frin_processed.npy"
if os.path.exists(frin_file):    
    newFRIN = np.load(frin_file)
else:
    #fp = input_path + "WAD2M_wetlands_2000-2020_05deg_Ver2.0.nc"    
    fp = input_path + "WAD2M_wetlands_2000-2020_025deg_Ver2.0.nc"    
    year_range = end_year-start_year+1
    newFRIN = []
    for year in range(2006,2018+1):
        print(year)
        newFRIN.append(load_newFRIN(fp, year))
    #
    newFRIN = np.concatenate(newFRIN, axis=-1)
    np.save(frin_file, newFRIN)
print(newFRIN.shape)  # (720, 360, 156)
print(np.nanmean(newFRIN))

### offset southern hemisphere by 6 months
shifted = newFRIN[:, 0:180, shift_size:].copy()  # new var of shifted data
newFRIN[:, 0:180, :] = np.nan  # replace old data with nans
newFRIN[:, 0:180, 0:-shift_size] = shifted.copy()  # shove in shifted data
print(newFRIN.shape)  # (720, 360, 156)
print(np.nanmean(newFRIN))

# # plot for sanity check
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# import matplotlib.colors as colors
# from matplotlib.colors import SymLogNorm
# data = newFRIN
# data = np.squeeze(data)
# data[data ==-9999] = np.nan
# data = np.nanmean(data, axis=-1)  # ********************** CORRECT INDEX???????????*****************
# plt.hist(data.flatten(), bins=100)
# plt.show()
# data = data.T
# lon = np.linspace(-180, 180, 720)
# lat = np.linspace(-90, 90, 360)
# fig = plt.figure(figsize=(11, 6))
# ax = plt.axes(projection=ccrs.PlateCarree())    
# heatmap = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
#                         cmap='coolwarm')
# ax.coastlines()  # Outline of continents
# ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', alpha=0.3)
# ax.add_feature(cfeature.OCEAN, facecolor='white', alpha=0.2)
# ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)
# cbar = fig.colorbar(heatmap, ax=ax, orientation='horizontal',
#                     pad=0.05,  # distance from plot
#                     fraction=0.035,  # relative width of colorbar
#                     shrink=0.7,  # shrink height (for horizontal, this affects thickness)
#                     aspect=25)  # ratio of long to short axis
# plt.tight_layout()
# plt.show()


# In[67]:


def yearly(timeseries):
    days_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    tot = 0
    for month in range(12):
        if not np.isnan(timeseries[month]):  # skip nans
            tot += timeseries[month] * days_per_month[month]
    return tot


# In[68]:


### separate function to retain monthly estimates (Youmi asked for this)

@torch.no_grad()
def youmi_upscale(sitedata, area, frin, I_obs, lat):
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
            # print(x_piece.shape)
            # print(i_piece.shape)
            I_input = torch.unsqueeze(i_piece, dim=0).to(torch.float32)
            # print(i_piece.shape)
            # print(I_input.shape, X_input.shape, hidden.shape)
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


# In[69]:


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


# In[70]:


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


# In[71]:


### subset the sites for parallel computing
shuffled_indices = np.arange(num_sites)  # shuffle—consistently across reps using same random seed                                                                                                                                                 
rng = np.random.default_rng(seed=123)
rng.shuffle(shuffled_indices)

# subset table for current rep                                                                                                                                                                            
max_concurrent_requests = 1000
n = int(np.ceil(num_sites / max_concurrent_requests))   # 50788/1000=50.788
print(n)


# In[72]:


### the loop: predict on each grid cell  

#for i in range(run*n, (run*n)+n):  # production version
for i in range(2):  # testing version
    if i < num_sites:  # not all i's exist using range(it*n, (it*n)+n)                                                                                                                                                              
        site = shuffled_indices[i]
        # site_output_fp = ml_output_path + "/production_estimates_site_" + str(site) + ".npy"
        site_output_fp = ml_output_path + "/youmi_estimates_site_" + str(site) + ".npy"
        if os.path.exists(site_output_fp):    
            print(site_output_fp, "exists")
        else:
            long,lat = longlat[site]
            long_ind,lat_ind = coords2index(long, lat) 
            
            # load MODIS image
            if site in match_map:   # ************************************ skips some sites ************************          
                folder = modis_path + "/site_" + str(match_map[site]) + "/"
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
                                
                youmi_estimates = []
                #for rep in range(100):  # production
                for rep in range(2):  # testing
                    print("rep", rep, "site", site)
                
                    # load up model
                    fp = final_model_path + "/production_rep_" + str(rep) + ".sav"
                    checkpoint=torch.load(fp, map_location=torch.device('cpu'), weights_only=False)
                    model=pureML_GRU(7,n_a,n_l,1,dropout)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(device)
                    model.eval()    
                        
                    # tile across the big image
                    tiles = []
                    youmi_tiles = []
                    tile_size = 10  # model expects 10x10 images
                    nrows = int(np.floor(I.shape[-2]/tile_size))  
                    ncols = int(np.floor(I.shape[-1]/tile_size))
                    for row in range(nrows):
                        for col in range(ncols):
                            y = row*tile_size
                            x = col*tile_size
                            tile = I[:,:,y:y+tile_size,x:x+tile_size]  # torch.Size([156, 7, 10, 10])
                            youmi_tile = youmi_upscale(X_sim[site], 
                                                       area_sim[site], 
                                                       newFRIN[long_ind,lat_ind,:], 
                                                       tile, 
                                                       lat)  # torch.Size([12, 12])
                            youmi_tiles.append(youmi_tile)
                
                    # average across tiles (we're still inside a single rep)
                    youmi_tiles = np.stack(youmi_tiles, axis=-1)  # (12, 12, 121)
                    youmi_tiles = np.nanmean(youmi_tiles, axis=-1)  # (12, 12)
                    youmi_estimates.append(youmi_tiles)
                
                # save
                youmi_estimates = np.stack(youmi_estimates, axis=-1)  # (12, 12, 100)
                print(youmi_estimates.shape)
                # np.save(site_output_fp, estimates)
                # site_output_fp = ml_output_path + "/youmi_estimates_site_" + str(site)
                np.save(site_output_fp, youmi_estimates)


# In[ ]:





# In[ ]:





# In[ ]:





# In[112]:


### plot outputs from forward pass with different temperatures

i=0
site = shuffled_indices[i]
site_output_fp = ml_output_path + "/youmi_estimates_site_" + str(site) + ".npy"
long,lat = longlat[site]
long_ind,lat_ind = coords2index(long, lat) 
print(lat, long)

temper_idx = 1  # ['LE' 'temp' 'site_class_1' 'site_class_2' 'site_class_3' 'site_class_4']
X_input = X_sim[site].clone()  # isolate site  torch.Size([156, 7])
X_input[X_input==-9999] = np.nan  # convert to nan
mean = np.nanmean(X_input[:,temper_idx])
X_input[:,temper_idx] -= mean  # center temperature data at 0.0
X_input = np.nan_to_num(X_input, nan=-9999)  # replace nans again
X_input = torch.tensor(X_input)

# load MODIS image
folder = modis_path + "/site_" + str(match_map[site]) + "/"
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
                
rep=0

# load up model
fp = final_model_path + "/production_rep_" + str(rep) + ".sav"
checkpoint=torch.load(fp, map_location=torch.device('cpu'), weights_only=False)
model=pureML_GRU(7,n_a,n_l,1,dropout)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()    
    
# tile across the big image
tiles = []
youmi_tiles = []
tile_size = 10  # model expects 10x10 images
nrows = int(np.floor(I.shape[-2]/tile_size))  
ncols = int(np.floor(I.shape[-1]/tile_size))
row,col=0,0
# for row in range(nrows):
#     for col in range(ncols):
y = row*tile_size
x = col*tile_size
tile = I[:,:,y:y+tile_size,x:x+tile_size]  # torch.Size([156, 7, 10, 10])

# pred
preds = []
for i in range(50):
    X_input[:,temper_idx] += 1  # add 1 degree C
    output = youmi_upscale(X_input, 
                               area_sim[site], 
                               newFRIN[long_ind,lat_ind,:], 
                               tile, 
                               lat)  # torch.Size([12, 12])
    #
    # output = output.flatten()
    # plt.plot(output)
    # plt.show()
    preds.append(output.nanmean(axis=0).nansum())

# plot
fig = plt.gcf()
fig.set_size_inches(4, 3)  # width, height in inches
plt.plot(preds)
plt.xlabel("Mean temperature (C), artificially adjusted")
plt.ylabel("Emission (Tg/yr/m^2)")
plt.title(r"Individual grid cell: -7$^\circ$S 21$^\circ$E")
plt.savefig(ml_output_path + "/emission_v_temperature.pdf", bbox_inches="tight")
plt.show()


# In[ ]:




