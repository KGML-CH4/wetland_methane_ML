#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from copy import deepcopy


# In[2]:


def mse_missing(pred, true):
    mask = torch.where(true == -9999, 0, 1)  # 0/1   (e.g. a = torch.tensor([5,-999,3]); torch.where(a == -999, 0, 1))
    non_missing = torch.sum(mask, dim=1)  # sum row-wise: the count of non-missing timepoints per site/training example
    mse = (true-pred)**2  # squared error. Random values for missing data (error = pred-999)
    mse = mse * mask  # missing = 0
    mse = torch.sum(mse, dim=1)  # sum row-wise (by training example)
    mse /= non_missing  # MSE, by training example
    loss = mse.mean()  # mean across training examples
    return loss

def r2_missing(pred, true):
    y = np.array(true.clone().cpu())
    y[y == -9999] = float(np.nan)   # replace 999 with nan  
    mask = np.where(~np.isnan(np.array(y)))
    var_y = np.var(y[mask])
    square_errors = np.square(pred[mask].cpu()-y[mask])
    mse = square_errors.mean()
    return 1.0 - mse / var_y

# def r2_missing(pred, true):
#     y = true.clone().detach()
#     y[y == -9999] = float(np.nan)   # replace 999 with nan  
#     var_y = np.nanvar(y, ddof=False)
#     square_errors = np.square(pred-y)
#     mse = square_errors.nanmean()
#     return 1.0 - mse / var_y

# def r3_missing(pred, true):
#     y = true.clone().detach()
#     y[y == -9999] = float(np.nan)   # replace 999 with nan  
#     ssr = np.square(pred-y)
#     ssr = ssr.nanmean()  # https://en.wikipedia.org/wiki/Coefficient_of_determination
#     sst = np.square(y-y.nanmean())
#     sst = sst.nanmean()
#     return 1.0 - ssr / sst

class R2Loss(nn.Module):
    #calculate coefficient of determination
    def forward(self, y_pred, y):
        var_y = torch.var(y, unbiased=True)
        return 1.0 - F.mse_loss(y_pred, y, reduction="mean") / var_y

def get_gpu_memory():
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
  ACCEPTABLE_AVAILABLE_MEMORY = 1024
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
  memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
  memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
  print(memory_free_values)
  return memory_free_values

def Z_norm_reverse(X,Xnorm,units_convert=1.0):
    return (X*Xnorm[1]+Xnorm[0])*units_convert

def random_flip_rotate(image):
    if random.random() > 0.5:
        image = torch.flip(image, dims=[2])
    if random.random() > 0.5:
        image = torch.flip(image, dims=[1])
    k = random.randint(0, 3)
    image = torch.rot90(image, k=k, dims=[1, 2])
    return image

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


# In[3]:


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


# In[4]:


### file paths
model_version = "161"
TEM_preprocess_path = '/scratch.global/chriscs/KGML/Out/Emissions/fluxnet_sim_v1.sav'
MODIS_path = '/scratch.global/chriscs/KGML/Data/Emissions/MODIS_tiles_TEM/Preprocessed_tiles/'

input_path = '/scratch.global/chriscs/KGML/Data/Emissions/'
output_path = '/scratch.global/chriscs/KGML/Out/Emissions/v' + model_version + "/"
# input_path = '/Users/chris/TempWorkSpace/KGML/Data/Emissions/'
# output_path = '/Users/chris/TempWorkSpace/KGML/Out/Emissions/v' + model_version + "/"

path_save_obs = output_path + 'fluxnet_obs_v' + model_version + '.sav'
path_save_sim = output_path + 'fluxnet_sim_v' + model_version + '.sav'
pretrain_path = output_path + 'train_' + str(model_version) + '.sav'
finetune_path = output_path + 'finetune_' + str(model_version) + '.sav'
#path_shuffle = output_path + 'shuffle_inds_v' + model_version + '.sav'


# In[5]:


### params
start_year, end_year = 2006, 2018  # these are the years of fluxnet data
num_years=end_year-start_year+1
timesteps_per_year=12
timesteps=timesteps_per_year*num_years
num_windows=int(np.floor((timesteps-24)/12+1))   # ***DON'T TOUCH—model does not automatically scale to different window sizes***
days_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]
print(num_windows)
nonmissing_required = 4


# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


### load observed data first (b/c want to filter these sites from TEM training)
data0 = torch.load(path_save_obs, weights_only=False)
X_obs = data0['X']
Y_obs = data0['Y']
Z_obs = data0['Z']
I_obs = data0['I']
print(X_obs.shape)
print(Y_obs.shape)
print(Z_obs.shape)
print(I_obs.shape)
X_vars_obs = data0['X_vars']
Y_vars_obs = data0['Y_vars']
print(Y_vars_obs, flush=True)
Z_vars_obs = data0['Z_vars']
X_stats = data0['X_stats']
Y_stats = data0['Y_stats']
I_stats = data0['I_stats']
print(X_vars_obs, flush=True)
#print(Y_vars, flush=True)
print(Z_vars_obs, flush=True)

#Z = Z[:,:,1:3]  # just lat, long to match formatting of sims

# site ID 
ids = np.arange(X_obs.shape[0])  # this notebook is splitting by site ID—no option for random split
print(len(set(list(ids))), "sites")


# In[7]:


### Separate out F_CH4 (into variable "M")

# create new var
mind = list(X_vars_obs).index("FCH4")
M_obs = deepcopy(X_obs[:,:,mind])
M_vars_obs = deepcopy(X_vars_obs[mind])
M_stats = deepcopy(X_stats[mind, :])
print(M_vars_obs, M_obs.shape, M_stats)

# # change -9999 to nan
# M_obs[M_obs == -9999] = np.nan

# # unnormalize
# M_obs = Z_norm_reverse(M_obs, X_stats[mind,:])

# remove from X
X_obs = np.delete(X_obs, mind, axis=2)
X_vars_obs = np.delete(X_vars_obs, mind)
X_stats = np.delete(X_stats, mind, axis=0)

print(X_obs.shape)
print(X_vars_obs)
print(X_stats.shape)


# In[8]:


### Separate out FCH4_F_ANNOPTLM (into variable "G")

# create new var
gind = list(X_vars_obs).index("FCH4_F_ANNOPTLM")
G_obs = X_obs[:,:,gind]
G_vars_obs = X_vars_obs[gind]
G_stats = deepcopy(X_stats[gind, :])
print(G_vars_obs, G_obs.shape, G_stats)

# remove from X
X_obs = np.delete(X_obs, gind, axis=2)
X_vars_obs = np.delete(X_vars_obs, gind)
X_stats = np.delete(X_stats, gind, axis=0)

print(X_obs.shape)
print(X_vars_obs)
print(X_stats.shape)


# In[9]:


### prep and filter time windows
print(X_obs.shape)
print(Y_obs.shape)
print(Z_obs.shape)
print(M_obs.shape)
print(G_obs.shape)
print(I_obs.shape)
 
# chunk up train sites into windows
new_X, new_Y, new_Z, new_M, new_G, new_I = [],[],[],[],[],[]
num_sites = X_obs.shape[0]
for site in range(num_sites):
    new_site_x, new_site_y, new_site_z, new_site_m, new_site_g, new_site_i = [],[],[],[],[],[]
    for it in range(num_windows):
        window_range_in = range(timesteps_per_year*it,timesteps_per_year*it+(timesteps_per_year*2))  # window of (smaller) input
        x_piece = X_obs[site, window_range_in, :]#.to(device)
        y_piece = Y_obs[site, window_range_in, 0]#.to(device)
        z_piece = Z_obs[site, window_range_in, :]#.to(device)
        m_piece = M_obs[site, window_range_in]#.to(device)
        g_piece = G_obs[site, window_range_in]#.to(device)
        i_piece = I_obs[site, window_range_in]#.to(device)
        # print(x_piece.shape, y_piece.shape, z_piece.shape)  # torch.Size([24, 6]) torch.Size([24]) torch.Size([24, 3])

        # year 1
        x_piece_1 = x_piece[0:timesteps_per_year, :]
        y_piece_1 = y_piece[0:timesteps_per_year]
        i_piece_1 = y_piece[0:timesteps_per_year]

        # first check: are inputs consistently missing for each time step? I would it expect the model to get tripped up otherwise.
        # (rather, instead of checking, just assigning missing to all inputs where there is at least one missing)
        x_missing_1 = (x_piece_1 == -9999).any(dim=1)  # check which months have missing inputs for any variable
        for month in range(timesteps_per_year):
            if x_missing_1[month] == True: # if any is missing, replace all with missing
                x_piece[month, :] = -9999  # note this is the "full" x_piece that gets passed through
                i_piece[month] = -9999                        

        # (skipping filter on missing data in year-1, i.e., it can be all missing)
                
        # year 2
        x_piece_2 = x_piece[timesteps_per_year:(timesteps_per_year*2), :]
        y_piece_2 = y_piece[timesteps_per_year:(timesteps_per_year*2)]
        i_piece_2 = i_piece[timesteps_per_year:(timesteps_per_year*2)]
        
        # second check: do positions of missing x values == missing y value positions?
        # We cannot expect the model to output a good estimate for, say, January, if there are no inputs.
        # But the reciprocal: we should keep the inputs even if no output, because that's more information for the RNN to work with.
        x_missing_2 = (x_piece_2 == -9999).any(dim=1)
        y_missing_2 = (y_piece_2 == -9999) 
        for month in range(timesteps_per_year):
            if x_missing_2[month] == True:
                x_piece_2[month, :] = -9999  # if any X's are missing, set the rest to missing to avoid confusing the model
                y_piece_2[month] = -9999
                x_piece[month + timesteps_per_year, :] = -9999  # (+timesteps_per_year because we're modifying second year of the "full" x_piece)
                y_piece[month + timesteps_per_year] = -9999
                i_piece[month + timesteps_per_year, :] = -9999  
        # 
        x_missing_2 = (x_piece_2 == -9999).any(dim=1)  # re-counting after adding nans
        y_missing_2 = (y_piece_2 == -9999)  

        # third check: do we have enough months with non-missing data in year 2?
        if torch.sum(y_missing_2) <= (timesteps_per_year-nonmissing_required):
            #print(np.sum(np.array(np.isnan(y_piece))))
            #y_piece = torch.nan_to_num(y_piece, nan=-999)  # replace nan with -999 for the custom loss
            y_piece = np.expand_dims(y_piece, axis=-1)
            #print(x_piece.shape, y_piece.shape, z_piece.shape)
            new_site_x.append(x_piece)
            new_site_y.append(y_piece)
            new_site_z.append(z_piece)
            new_site_m.append(m_piece)
            new_site_g.append(g_piece)
            new_site_i.append(i_piece)

    # add site to 4d list
    if len(new_site_x) > 0:
        print("site", site, ", num windows", len(new_site_x))
    else:
        print("site", site, ", num windows", len(new_site_x), "*")
    new_X.append(torch.tensor(np.array(new_site_x)))
    new_Y.append(torch.tensor(np.array(new_site_y)))
    new_Z.append(torch.tensor(np.array(new_site_z)))
    new_M.append(torch.tensor(np.array(new_site_m)))
    new_G.append(torch.tensor(np.array(new_site_g)))
    new_I.append(torch.tensor(np.array(new_site_i)))

X_obs_windows = list(new_X)
Y_obs_windows = list(new_Y)
Z_obs_windows = list(new_Z)
M_obs_windows = list(new_M)
G_obs_windows = list(new_G)
I_obs_windows = list(new_I)
print(len(X_obs_windows), len(Y_obs_windows), len(Z_obs_windows), len(M_obs_windows), len(G_obs_windows), len(I_obs_windows))
print(X_obs_windows[0].shape, Y_obs_windows[0].shape, Z_obs_windows[0].shape, M_obs_windows[0].shape, G_obs_windows[0].shape, I_obs_windows[0].shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[11]:


### load simulated data
data0 = torch.load(TEM_preprocess_path, weights_only=False)
X_sim = data0['X']
Y_sim = torch.tensor(data0['Y'])
Z_sim = data0['Z']
X_vars_sim = data0['X_vars']
Y_vars_sim = data0['Y_vars']
Z_vars_sim = data0['Z_vars']
X_stats_sim = data0['X_stats']
Y_stats_sim = data0['Y_stats']
print(Y_stats_sim)

print(X_vars_sim, flush=True)
print(Y_vars_sim, flush=True)
print(Z_vars_sim, flush=True)
print(X_sim.shape, flush=True)
print(Y_sim.shape, flush=True)
print(Z_sim.shape, flush=True)


# # filter for say 1000 sites for now                  ****************************************************************************************************************
# n=1000
# inds = np.random.choice(np.random.randint(X.shape[0]), size=n, replace=False)
# X_sim = X_sim[inds]
# Y_sim = Y_sim[inds]
# Z_sim = Z_sim[inds]
# print(X_sim.shape, flush=True)
# print(Y_sim.shape, flush=True)
# print(Z_sim.shape, flush=True)
# print(np.nanmin(X_sim), np.nanmax(X_sim))
# print(np.nanmin(Y_sim), np.nanmax(Y_sim))

# site ID 
ids = np.arange(X_sim.shape[0])  # this notebook is splitting by site ID—no option for random split
print(len(set(list(ids))), "sites")


# In[12]:


### filter eddy covariance sites from TEM

print(X_sim.shape)
print(Y_sim.shape)
print(Z_sim.shape)

# get obs grid cells
bad_cells = {}
num_sites = X_obs.shape[0]
for site in range(num_sites):
    # print(site)
    lat_ind = list(Z_vars_obs).index("LAT")
    long_ind = list(Z_vars_obs).index("LON")
    lat = np.nanmax(Z_obs[site,:,lat_ind])
    long = np.nanmax(Z_obs[site,:,long_ind])
    long,lat = coords2index(long, lat)
    coords = str(long) + "_" + str(lat)
    if coords in bad_cells:
        print("repeat:", lat, long)
    else:
        bad_cells[coords] = 0

# loop through TEM sites and make sure they don't overlap with flux towers
new_X = []
new_Y = []
new_Z = []
num_sites = X_sim.shape[0]
good_sites_sim = []
for site in range(num_sites):
    lat_ind = list(Z_vars_sim).index("lat")
    long_ind = list(Z_vars_sim).index("long")
    lat = np.nanmax(Z_sim[site,:,lat_ind])
    long = np.nanmax(Z_sim[site,:,long_ind])
    long,lat = coords2index(long, lat)
    coords = str(long) + "_" + str(lat)
    
    if coords in bad_cells:
        print("match", lat, long)
    else:
        new_X.append(X_sim[site])
        new_Y.append(Y_sim[site])
        new_Z.append(Z_sim[site])
        good_sites_sim.append(site)
#
X_sim = torch.tensor(np.array(new_X))
Y_sim = torch.tensor(np.array(new_Y))
Z_sim = torch.tensor(np.array(new_Z))
print(X_sim.shape)
print(Y_sim.shape)
print(Z_sim.shape)


# In[13]:


# ### prep and filter time windows
# print(X_obs.shape)
# print(Y_obs.shape)
# print(Z_obs.shape)
 
# # chunk up train sites into windows
# windowed_indices_sim = {}
# new_X, new_Y, new_Z = [],[],[]
# num_sites = X_sim.shape[0]
# for site in range(num_sites):
#     windowed_indices_sim[site] = []
#     if site % 1000 == 0:
#         print(site)
#     new_site_x, new_site_y, new_site_z = [],[],[]
#     for it in range(num_windows):
#         window_range_in = range(timesteps_per_year*it,timesteps_per_year*it+(timesteps_per_year*2))  # window of (smaller) input
#         x_piece = X_sim[site, window_range_in, :]#.to(device)
#         y_piece = Y_sim[site, window_range_in, 0]#.to(device)
#         z_piece = Z_sim[site, window_range_in, :]#.to(device)
#         # print(x_piece.shape, y_piece.shape, z_piece.shape)  # torch.Size([24, 6]) torch.Size([24]) torch.Size([24, 3])

#         # year 1
#         x_piece_1 = x_piece[0:timesteps_per_year, :]
#         y_piece_1 = y_piece[0:timesteps_per_year]

#         # first check: are inputs consistently missing for each time step? I would it expect the model to get tripped up otherwise.
#         # (rather, instead of checking, just assigning missing to all inputs where there is at least one missing)
#         x_missing_1 = (x_piece_1 == -9999).any(dim=1)  # check which months have missing inputs for any variable
#         for month in range(timesteps_per_year):  # timesteps 1-12
#             if x_missing_1[month] == True: # if any is missing, replace all with missing
#                 x_piece[month, :] = -9999  # note this is the "full" x_piece that gets passed through
                
#         # (skipping filter on missing data in year-1, i.e., it can be all missing)
                
#         # year 2
#         x_piece_2 = x_piece[timesteps_per_year:(timesteps_per_year*2), :]
#         y_piece_2 = y_piece[timesteps_per_year:(timesteps_per_year*2)]
        
#         # second check: do positions of missing x values == missing y value positions?
#         # We cannot expect the model to output a good estimate for, say, January, if there are no inputs.
#         # But the reciprocal: we should keep the inputs even if no output, because that's more information for the RNN to work with.
#         x_missing_2 = (x_piece_2 == -9999).any(dim=1)
#         y_missing_2 = (y_piece_2 == -9999) 
#         for month in range(timesteps_per_year):
#             if x_missing_2[month] == True:
#                 x_piece_2[month, :] = -9999  # if any X's are missing, set the rest to missing to avoid confusing the model
#                 y_piece_2[month] = -9999
#                 x_piece[month + timesteps_per_year, :] = -9999  # (+timesteps_per_year because we're modifying second year of the "full" x_piece)
#                 y_piece[month + timesteps_per_year] = -9999
#         # 
#         x_missing_2 = (x_piece_2 == -9999).any(dim=1)  # re-counting after adding nans
#         y_missing_2 = (y_piece_2 == -9999)  
        
#         # third check: do we have enough months with non-missing data in year 2?
#         if torch.sum(y_missing_2) <= (timesteps_per_year-nonmissing_required):
#             #print(np.sum(np.array(np.isnan(y_piece))))
#             #y_piece = torch.nan_to_num(y_piece, nan=-999)  # replace nan with -999 for the custom loss
#             y_piece = np.expand_dims(y_piece, axis=-1)
#             #print(x_piece.shape, y_piece.shape, z_piece.shape)
#             new_site_x.append(x_piece)
#             new_site_y.append(y_piece)
#             new_site_z.append(z_piece)
#             windowed_indices_sim[site].append(it)

#     # add site to 4d list
#     if len(new_site_x) > 0:        
#         new_X.append(torch.tensor(np.array(new_site_x)))
#         new_Y.append(torch.tensor(np.array(new_site_y)))
#         new_Z.append(torch.tensor(np.array(new_site_z)))

# X_sim_windows = list(new_X)
# Y_sim_windows = list(new_Y)
# Z_sim_windows = list(new_Z)
# print(len(X_sim_windows), len(Y_sim_windows), len(Z_sim_windows))
# print(X_sim_windows[0].shape, Y_sim_windows[0].shape, Z_sim_windows[0].shape)


# In[14]:


# ### save intermediate
# torch.save({'X_sim_windows': X_sim_windows,
#             'Y_sim_windows': Y_sim_windows,
#             'Z_sim_windows': Z_sim_windows,
#             'X_vars': X_vars_sim,
#             'windowed_indices_sim': windowed_indices_sim,
#             }, path_save_sim + "_windows")

# load
data0 = torch.load(path_save_sim + "_windows", weights_only=False)
X_sim_windows = data0['X_sim_windows']
Y_sim_windows = data0['Y_sim_windows']
Z_sim_windows = data0['Z_sim_windows']
X_vars_sim = data0['X_vars']
windowed_indices_sim = data0['windowed_indices_sim']
print(len(X_sim_windows))
print(len(Y_sim_windows))
print(len(Z_sim_windows))
print(X_vars_sim)
print(len(X_sim_windows))
print(X_sim_windows[0].shape)
print(Y_sim_windows[0].shape)
print(Z_sim_windows[0].shape)
del data0
import gc
gc.collect()


# In[15]:


### train/val/test split
def split_data_group(data0,shuffled_ind,train_frac=0.7,val_frac=0.3):
    sample_size = len(data0)
    train_n=int(train_frac*sample_size)
    val_n=int(val_frac*sample_size)
    test_n=sample_size - train_n -val_n
    data_train, data_val, data_test = [],[],[]
    for i in range(train_n):
        data_train.append(data0[shuffled_ind[i]])
    for i in range(train_n, train_n+val_n):
        data_val.append(data0[shuffled_ind[i]])
    for i in range(train_n+val_n, sample_size):
        data_test.append(data0[shuffled_ind[i]])
    data_train = torch.cat(data_train, dim=0)
    data_val = torch.cat(data_val, dim=0)
    data_test = torch.cat(data_test, dim=0)
    # data_train = torch.tensor(np.array(data_train))
    # data_val = torch.tensor(np.array(data_val))
    # data_test = torch.tensor(np.array(data_test))
    data_train = data_train.to(torch.float32)
    data_val = data_val.to(torch.float32)
    data_test = data_test.to(torch.float32)
    return data_train, data_val, data_test, 

# shuffle remaining
X_temp = deepcopy(X_sim_windows)
Y_temp = deepcopy(Y_sim_windows)
Z_temp = deepcopy(Z_sim_windows)
print(X_temp[0].shape)  # torch.Size([156, 6])
# del X_temp[test_ind]
# del Y_temp[test_ind]
# del Z_temp[test_ind]
shuffled_ind = torch.randperm(len(X_temp))
# path_shuffle = output_path + 'shuffle_inds_site_' + str(test_ind) + "_rep_" + str(rep) + '.sav'
# print(path_shuffle)
# if not os.path.exists (path_shuffle):
#     print("whipping up new shuffled indices")
#     shuffled_ind = torch.randperm(len(X_temp))
#     torch.save({'shuffled_ind':shuffled_ind
#                },path_shuffle)
# else:
#     print("using existing shuffled indices")
#     tmp = torch.load(path_shuffle, weights_only=False)
#     shuffled_ind = tmp['shuffled_ind']
# #print(shuffled_ind, flush=True)

# X
X_train_sim, X_val_sim, X_test_sim = split_data_group(X_temp,shuffled_ind)
print(X_train_sim.size(), X_val_sim.size(), X_test_sim.size(), flush=True)

# Y
Y_train_sim, Y_val_sim, Y_test_sim = split_data_group(Y_temp,shuffled_ind)
print(Y_train_sim.size(), Y_val_sim.size(), Y_test_sim.size(),flush=True)

# Z
Z_train_sim, Z_val_sim, Z_test_sim = split_data_group(Z_temp,shuffled_ind)
print(Z_train_sim.size(), Z_val_sim.size(), Z_test_sim.size(),flush=True)


# In[16]:


### initialize params for training
n_a=8 #hidden state number
n_l=2 #layer of gru
starttime=time.time()
loss_val_best = 500000
R2_best=0.5
best_epoch = 1000
lr_adam=0.001 #orginal 0.0001
bsz = 1000
train_losses = []
val_losses = []


# In[17]:


### initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device, flush=True)

dropout = 0
model = pureML_GRU(len(X_vars_obs),n_a,n_l,1,dropout)
model.to(device)

print(model, flush=True)
params = list(model.parameters())
print(len(params), flush=True)
print(params[5].size(), flush=True)  # conv1's .weight
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size(), flush=True)
    
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr_adam) #add weight decay normally 1-9e-4

# scheduler
# decay_time = 80  # og=80
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_time, gamma=0.5)
# maxepoch=decay_time*4
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
maxepoch=100


# In[18]:


def load_tile(epoch):
    print("loading tile", epoch)
    means = np.load(MODIS_path + "/global_means.npy")
    sds = np.load(MODIS_path + "/global_SDs.npy")
    tile = np.load(MODIS_path + "/tile_" + str(epoch) + ".npy")  # (12519, 156, 7, 10, 10)
    
    # normalize
    print("normalizing")
    means = means.reshape(1, 1, 7, 1, 1)
    tile = tile - means
    sds = sds.reshape(1, 1, 7, 1, 1)
    tile = tile / sds
    
    # remove fluxnet sites
    print("removing a few sites")
    tile = tile[good_sites_sim]  # (12499, 156, 7, 10, 10)
    
    # filter for approved windows
    print("prep windows")
    new_tile = []  # 12498
    for site in range(len(windowed_indices_sim)):
        new_site = []
        for it in windowed_indices_sim[site]:  # this skips any windows that aren't pre-approved
            window_range_in = range(timesteps_per_year*it,timesteps_per_year*it+(timesteps_per_year*2))
            new_site.append(tile[site, window_range_in, :, :, :])
        #
        if len(new_site) > 0:
            new_tile.append(torch.tensor(np.array(new_site)))
    #
    del tile
    gc.collect()
    
    # train val test split
    print("train val test split")
    train_frac=0.7; val_frac=0.3
    sample_size = len(new_tile)
    train_n=int(train_frac*sample_size)
    val_n=int(val_frac*sample_size)
    #test_n=sample_size - train_n -val_n
    I_train_sim, I_val_sim, I_test_sim = [],[],[]
    for i in range(train_n):
        I_train_sim.append(new_tile[shuffled_ind[i]])
    for i in range(train_n, train_n+val_n):
        I_val_sim.append(new_tile[shuffled_ind[i]])
    # for i in range(train_n+val_n, sample_size):
    #     I_test_sim.append(new_tile[shuffled_ind[i]])
    I_train_sim = torch.cat(I_train_sim, dim=0)
    I_val_sim = torch.cat(I_val_sim, dim=0)
    #I_test_sim = torch.cat(I_test_sim, dim=0)
    I_train_sim = I_train_sim.to(torch.float32)
    I_val_sim = I_val_sim.to(torch.float32)
    # I_test_sim = I_test_sim.to(torch.float32)
    del new_tile
    gc.collect()
    
    print(I_train_sim.size(), I_val_sim.size(), flush=True)    
    return I_train_sim, I_val_sim


# In[19]:


### pretrain
train_n = X_train_sim.size(0)
val_n = X_val_sim.size(0)
test_n = X_test_sim.size(0)
X_val = X_val_sim.to(device)
Y_val = Y_val_sim.to(device)
for epoch in range(maxepoch):

    # load image data
    I_train_sim, I_val_sim = load_tile(epoch)
    I_val_sim = I_val_sim.to(device)
    
    train_loss=0.0
    val_loss=0.0
    shuffled_b=torch.randperm(X_train_sim.size(0)) 
    X_train_shuff=X_train_sim[shuffled_b,:,:] 
    Y_train_shuff=Y_train_sim[shuffled_b,:,:]
    I_train_shuff=I_train_sim[shuffled_b,:,:]
    
    # forward
    model.train()  # (switch on dropout; and optimization?)
    model.zero_grad()
    for bb in range(int(train_n/bsz)):
        if bb != int(train_n/bsz)-1:
            sbb = bb*bsz
            ebb = (bb+1)*bsz
        else:
            sbb = bb*bsz
            ebb = train_n
        hidden = model.init_hidden(ebb-sbb).to(device)
        optimizer.zero_grad()
        X_input = X_train_shuff[sbb:ebb, :, :].to(device)
        Y_true = Y_train_shuff[sbb:ebb, :, :].to(device)
        I_input = I_train_shuff[sbb:ebb].to(device)

        #print(np.sum(np.array(np.isnan(X_input))))
        #print(np.sum(np.array(np.isnan(Y_true))))
        Y_est, _ = model(I_input, X_input, hidden)

        Y_est = Y_est[:,timesteps_per_year:(timesteps_per_year*2),:]  # chopping off the first year, that was spin-up
        Y_true = Y_true[:,timesteps_per_year:(timesteps_per_year*2),:]

        
        loss = mse_missing(Y_est, Y_true)
        # print(loss)
        # sys.exit()
        hidden.detach_()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item() * (ebb-sbb)  # (NOT "bsz", since some batches aren't full)
    #
    
    # validation
    model.eval()  # "testing" model, it switches off dropout and batch norm.    
    with torch.no_grad():

        # finalize training loss         
        train_loss /= train_n
        train_losses.append(train_loss)
        
        hidden = model.init_hidden(X_val.shape[0]).to(device)
        Y_val_pred_t, _ = model(I_val_sim, X_val, hidden)
        
        Y_val_pred_t = Y_val_pred_t[:,timesteps_per_year:(timesteps_per_year*2),:]  # chopping off the first year, that was spin-up
        
        loss = mse_missing(Y_val_pred_t, Y_val[:,timesteps_per_year:(timesteps_per_year*2),:])
        val_loss += loss.item() * timesteps_per_year * X_val.shape[0]
        #
        val_loss /= (val_n*timesteps_per_year)        
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        for param_group in optimizer.param_groups:
            print(f"Learning rate after epoch {epoch+1}: {param_group['lr']}")
        
        # r2 on validation set this time
        val_R2 = []
        for varn in range(1):
            # Ysim = Z_norm_reverse(Y_val_pred_t[:,:,varn],Y_stats[varn,:])
            # Yobs = Z_norm_reverse(Y_val[:,:,varn],Y_stats[varn,:])
            # val_R2.append(r2_missing(Ysim.contiguous().view(-1),Yobs.contiguous().view(-1)).item())
            #val_R2.append(r2_missing(Y_val_pred_t.contiguous().view(-1),Y_val.contiguous().view(-1)).item())
            val_R2.append(r2_missing(Y_val_pred_t.reshape(-1),Y_val[:,timesteps_per_year:(timesteps_per_year*2),:].reshape(-1)).item())
            
        # save model, update LR
        if val_loss < loss_val_best:
            loss_val_best=np.array(val_loss)
            R2_best = val_R2
            best_epoch = epoch
            #os.remove(pretrain_path)
            torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    #'R2': train_R2,
                    'loss': train_loss,
                    'los_val': val_loss,
                    #'R2_val': val_R2,
                    }, pretrain_path)   
        print("finished training epoch", epoch+1, flush=True)
        mtime=time.time()
        print("learning rate: ", optimizer.param_groups[0]["lr"], "train_loss: ", train_loss, "val_loss:",val_loss, "best_val_loss:",loss_val_best, "best_val_r2:",R2_best, f"Spending time: {mtime - starttime}s", flush=True)
        path_fs = pretrain_path+'fs'
        torch.save({'train_losses': train_losses,
                    'val_losses': val_losses,
                    'model_state_dict_fs': model.state_dict(),
                    }, path_fs)  
#
endtime=time.time()

print("final train_loss:",train_loss,"val_loss:",val_loss,"val loss best:",loss_val_best, flush=True)
print(f"total Training time: {endtime - starttime}s", flush=True)


# In[20]:


del I_train_sim, I_val_sim
gc.collect()


# In[43]:


### test to get TEM predictions
def check_results(total_b, check_xset, check_y1set, I_input):
    hidden = model.init_hidden(total_b)
    X_input = check_xset.to(device).float()
    Y_true = check_y1set.to(device)
    I_input = I_input.to(device).float()
    Y1_pred_t, _ = model(I_input, X_input, hidden)            
    print(I_input.shape, X_input.shape, Y1_pred_t.shape)
    return Y1_pred_t


outputs = []
with torch.no_grad():
    for test_ind in range(len(X_obs_windows)):
        print(test_ind)
        X_test = X_obs_windows[test_ind]
        Y_test = Y_obs_windows[test_ind]
        I_test = I_obs_windows[test_ind]
        test_n = len(X_test)
        checkpoint=torch.load(pretrain_path, map_location=torch.device('cpu'), weights_only=False)
        model=pureML_GRU(6,n_a,n_l,1,dropout)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device) #too large for GPU, kif not enough, change to cpu
        model.eval()  # this is "testing" model, it switches off dropout and batch norm.
        epoch = checkpoint['epoch']
        print("epoch", epoch, flush=True)
        Y_test_pred =  check_results(test_n, X_test, Y_test, I_test)    
        outputs.append(Y_test_pred)


# In[44]:


### process TEM predictions

for site in range(len(outputs)):

    # assign missing data where Y is missing (consistent with the other predictors in this model)
    missing = np.where(Y_obs_windows[site] == -9999)
    outputs[site][missing] = -9999

    # shove into X
    new_data = torch.cat([X_obs_windows[site], outputs[site]], dim=-1)
    X_obs_windows[site] = deepcopy(new_data)

# 
print(X_obs_windows[0].shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[45]:


### fine tune


# In[46]:


def split_data_group(data0,shuffled_ind,train_frac=0.7,val_frac=0.3):
    sample_size = len(data0)
    train_n=int(train_frac*sample_size)
    # val_n=int(val_frac*sample_size)
    # test_n=sample_size - train_n -val_n
    val_n = sample_size-train_n
    test_n=0
    data_train, data_val = [],[]
    for i in range(0, train_n):
        data_train.append(data0[shuffled_ind[i]])
    for i in range(train_n,sample_size):
        data_val.append(data0[shuffled_ind[i]])
    data_train = torch.cat(data_train, dim=0)
    data_val = torch.cat(data_val, dim=0)
    data_train = data_train.to(torch.float32)
    data_val = data_val.to(torch.float32)
    return data_train,data_val


# In[49]:


### train/val/test split

# separate single test site
###test_ind = 0
X_test = X_obs_windows[test_ind]
Y_test = Y_obs_windows[test_ind]
Z_test = Z_obs_windows[test_ind]
M_test = M_obs_windows[test_ind]
I_test = I_obs_windows[test_ind]
print(X_test.size(), Y_test.size(), Z_test.size(), M_test.size(), I_test.size(), flush=True)
if len(X_test) == 0:
    print("\n\n\n\t test data empty\n\n\n")
    sys.exit()

# shuffle remaining
X_temp = deepcopy(X_obs_windows)
Y_temp = deepcopy(Y_obs_windows)
Z_temp = deepcopy(Z_obs_windows)
M_temp = deepcopy(M_obs_windows)
I_temp = deepcopy(I_obs_windows)
del X_temp[test_ind]
del Y_temp[test_ind]
del Z_temp[test_ind]
del M_temp[test_ind]
del I_temp[test_ind]
# shuffled_ind = torch.randperm(len(X_temp))
path_shuffle = output_path + 'shuffle_inds_site_' + str(test_ind) + "_rep_" + str(rep) + '.sav'
print(path_shuffle)
if not os.path.exists (path_shuffle):
    print("whipping up new shuffled indices")
    shuffled_ind = torch.randperm(len(X_temp))
    torch.save({'shuffled_ind':shuffled_ind
               },path_shuffle)
else:
    print("using existing shuffled indices")
    tmp = torch.load(path_shuffle, weights_only=False)
    shuffled_ind = tmp['shuffled_ind']
#print(shuffled_ind, flush=True)

# X
X_train, X_val = split_data_group(X_temp,shuffled_ind)
print(X_train.size(), X_val.size(), flush=True)

# Y
Y_train, Y_val = split_data_group(Y_temp,shuffled_ind)
print(Y_train.size(), Y_val.size(), flush=True)

# Z
Z_train, Z_val = split_data_group(Z_temp,shuffled_ind)
print(Z_train.size(), Z_val.size(), flush=True)

# M
M_train, M_val = split_data_group(M_temp,shuffled_ind)
print(M_train.size(), M_val.size(), flush=True)

# I
I_train, I_val = split_data_group(I_temp,shuffled_ind)
print(I_train.size(), I_val.size(), flush=True)


# In[50]:


### initialize params for training
n_a=8 #hidden state number
n_l=2 #layer of gru
starttime=time.time()
loss_val_best = 500000
R2_best=0.5
best_epoch = 1000
lr_adam=0.001 #orginal 0.0001
bsz = 10
train_losses = []
val_losses = []


# In[56]:


### load pre-trained model 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device, flush=True)

dropout = 0
model = pureML_GRU(X_obs_windows[0].shape[-1],n_a,n_l,1,dropout)
model.to(device) #too large for GPU, kif not enough, change to cpu

print(model, flush=True)
params = list(model.parameters())
print(len(params), flush=True)
print(params[5].size(), flush=True)  # conv1's .weight
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size(), flush=True)

# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr_adam) #add weight decay normally 1-9e-4
# scheduler
# decay_time = 80  # og=80
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_time, gamma=0.5)
# maxepoch=decay_time*4
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
maxepoch=100


# In[57]:


### finetune
train_n = X_train.size(0)
val_n = X_val.size(0)
test_n = X_test.size(0)
X_val = X_val.to(device)
Y_val = Y_val.to(device)
I_val = I_val.to(device)

for epoch in range(maxepoch):
    train_loss=0.0
    val_loss=0.0
    shuffled_b=torch.randperm(X_train.size(0)) 
    X_train_shuff=X_train[shuffled_b,:,:] 
    Y_train_shuff=Y_train[shuffled_b,:,:]
    I_train_shuff=I_train[shuffled_b,:,:]
    
    # forward
    model.train()  # (switch on dropout; and optimization?)
    model.zero_grad()
    for bb in range(int(train_n/bsz)):
        if bb != int(train_n/bsz)-1:
            sbb = bb*bsz
            ebb = (bb+1)*bsz
        else:
            sbb = bb*bsz
            ebb = train_n
        hidden = model.init_hidden(ebb-sbb).to(device)
        optimizer.zero_grad()
        X_input = X_train_shuff[sbb:ebb, :, :].to(device)
        Y_true = Y_train_shuff[sbb:ebb, :, :].to(device)
        I_input = I_train_shuff[sbb:ebb].to(device)
        
        # augment images
        B, T, C, H, W = I_input.shape
        I_input = I_input.view(B * T, C, H, W)
        I_input = torch.stack([random_flip_rotate(img) for img in I_input])
        I_input = I_input.view(B, T, C, H, W)
        
        #print(np.sum(np.array(np.isnan(X_input))))
        #print(np.sum(np.array(np.isnan(Y_true))))
        #print(X_input)
        Y_est, _ = model(I_input, X_input, hidden)

        Y_est = Y_est[:,timesteps_per_year:(timesteps_per_year*2),:]  # chopping off the first year, that was spin-up
        Y_true = Y_true[:,timesteps_per_year:(timesteps_per_year*2),:]
        
        loss = mse_missing(Y_est, Y_true)
        # print(loss)
        # sys.exit()
        hidden.detach_()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item() * (ebb-sbb)  # (NOT "bsz", since some batches aren't full)
    #
    
    # validation
    model.eval()  # "testing" model, it switches off dropout and batch norm.    
    with torch.no_grad():

        # finalize training loss         
        train_loss /= train_n
        train_losses.append(train_loss)
        
        # val loss
        # for bb in range(int(val_n/bsz)):
        #     if bb != int(val_n/bsz)-1:
        #         sbb = bb*bsz
        #         ebb = (bb+1)*bsz
        #     else:
        #         sbb = bb*bsz
        #         ebb = val_n
        #     hidden = model.init_hidden(ebb-sbb)
        #     X_input = X_val[sbb:ebb, :, :].to(device)
        #     Y_true = Y_val[sbb:ebb, :, :].to(device)
        #     Y_val_pred_t, _ = model(X_input,hidden)
        #     loss = mse_missing(Y_val_pred_t, Y_true)
        #     val_loss += loss.item() * timesteps_per_year * (ebb-sbb)  # (NOT "bsz", since some batches aren't full)                
        hidden = model.init_hidden(X_val.shape[0]).to(device)
        Y_val_pred_t, _ = model(I_val, X_val, hidden)
        
        Y_val_pred_t = Y_val_pred_t[:,timesteps_per_year:(timesteps_per_year*2),:]  # chopping off the first year, that was spin-up
        
        loss = mse_missing(Y_val_pred_t, Y_val[:,timesteps_per_year:(timesteps_per_year*2),:])
        val_loss += loss.item() * timesteps_per_year * X_val.shape[0]
        #
        val_loss /= (val_n*timesteps_per_year)        
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        for param_group in optimizer.param_groups:
            print(f"Learning rate after epoch {epoch+1}: {param_group['lr']}")
        
        # r2 on validation set this time
        val_R2 = []
        for varn in range(1):
            # Ysim = Z_norm_reverse(Y_val_pred_t[:,:,varn],Y_stats[varn,:])
            # Yobs = Z_norm_reverse(Y_val[:,:,varn],Y_stats[varn,:])
            # val_R2.append(r2_missing(Ysim.contiguous().view(-1),Yobs.contiguous().view(-1)).item())
            #val_R2.append(r2_missing(Y_val_pred_t.contiguous().view(-1),Y_val.contiguous().view(-1)).item())
            val_R2.append(r2_missing(Y_val_pred_t.reshape(-1),Y_val[:,timesteps_per_year:(timesteps_per_year*2),:].reshape(-1)).item())
            
        # save model, update LR
        if val_loss < loss_val_best:
            loss_val_best=np.array(val_loss)
            R2_best = val_R2
            best_epoch = epoch
            #os.remove(finetune_path)
            torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    #'R2': train_R2,
                    'loss': train_loss,
                    'los_val': val_loss,
                    #'R2_val': val_R2,
                    }, finetune_path)   
        print("finished training epoch", epoch+1, flush=True)
        mtime=time.time()
        print("learning rate: ", optimizer.param_groups[0]["lr"], "train_loss: ", train_loss, "val_loss:",val_loss, "best_val_loss:",loss_val_best, "best_val_r2:",R2_best, f"Spending time: {mtime - starttime}s", flush=True)
        path_fs = finetune_path+'fs'
        torch.save({'train_losses': train_losses,
                    'val_losses': val_losses,
                    'model_state_dict_fs': model.state_dict(),
                    }, path_fs)  
#
endtime=time.time()

print("final train_loss:",train_loss,"val_loss:",val_loss,"val loss best:",loss_val_best, flush=True)
print(f"total Training time: {endtime - starttime}s", flush=True)


# In[59]:


### test
test_n = X_test.size(0)
def check_results(total_b, I_test, check_xset, check_y1set, Y_stats):
    print(check_y1set.shape)
    Y_true_all=torch.zeros((check_y1set.shape[0], timesteps_per_year, check_y1set.shape[2]))
    Y_pred_all=torch.zeros((check_y1set.shape[0], timesteps_per_year, check_y1set.shape[2]))    
    for bb in range(int(total_b/1)):
        if bb != int(total_b/1)-1:
            sbb = bb*1
            ebb = (bb+1)*1
        else:
            sbb = bb*1
            ebb = total_b
        hidden = model.init_hidden(ebb-sbb)
        X_input = check_xset[sbb:ebb, :, :].to(device)
        Y_true = check_y1set[sbb:ebb, :, :].to(device)
        I_input = I_test[sbb:ebb, :, :].to(device)
        Y1_pred_t, hidden = model(I_input, X_input,hidden)

        Y_true = Y_true[:,timesteps_per_year:(timesteps_per_year*2),:]  # chop off first year
        Y1_pred_t = Y1_pred_t[:,timesteps_per_year:(timesteps_per_year*2),:] 
        
        #            
        Y_true_all[sbb:ebb, :, :] = Y_true.to('cpu')  
        Y_pred_all[sbb:ebb, :, :] = Y1_pred_t.to('cpu')  
    #
    R2 = []
    loss = []    
    for varn in range(check_y1set.size(2)):
        loss.append(mse_missing(Y_pred_all[:,:,varn], Y_true_all[:,:,varn]).numpy())
        # Y_est = Z_norm_reverse(Y_pred_all[:,:,varn], Y_stats[varn,:])
        # Y_true = Z_norm_reverse(Y_true_all[:,:,varn],Y_stats[varn,:])
        # R2.append(r2_missing(Y_est.contiguous().view(-1),Y_true.contiguous().view(-1)).item())
        #R2.append(r2_missing(Y_pred_all.contiguous().view(-1),Y_true_all.contiguous().view(-1)).item())
    return Y_pred_all, R2, loss


with torch.no_grad():
    checkpoint=torch.load(finetune_path, map_location=torch.device('cpu'), weights_only=False)
    model=pureML_GRU(X_obs_windows[0].shape[-1],n_a,n_l,1,dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device) #too large for GPU, kif not enough, change to cpu
    model.eval()  # this is "testing" model, it switches off dropout and batch norm.
    epoch = checkpoint['epoch']
    print("epoch", epoch, flush=True)
    Y_test_pred,R_test,loss_test =  check_results(test_n, I_test.float(), X_test.float(), Y_test, Y_stats)    
    print(loss_test, R_test, flush=True)

    # write
    #Y_test_pred = list(Y_test_pred.flatten().numpy())
    path_out = output_path + 'output_site_' + str(test_ind) + "_rep_" + str(rep) + '.txt'
    print(path_out)
    with open(path_out, "w") as outfile:
        for window in range(Y_test_pred.shape[0]):
            outline = list(Y_test_pred[window,:,0].numpy())
            outline = "\t".join(list(map(str, outline)))
            outfile.write(outline + "\n")


# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


##### combine outputs  (after prep and filter windows)


# In[11]:


# initial organization
num_sites = len(Y_obs_windows)
print(num_sites)
num_reps = 100
X_temp = deepcopy(X_obs_windows)
FCH4 = deepcopy(M_obs_windows) 
FCH4_F_ANNOPTLM = deepcopy(G_obs_windows) 


# In[12]:


# load outputs
outputs = []
X_test = []
for site in range(num_sites):
    # print("site", site)
    new_site = []
    for rep in range(num_reps):
        new_rep = []
        path_out = output_path + 'output_site_' + str(site) + "_rep_" + str(rep) + '.txt'
        # print(path_out)
        
        if os.path.exists(path_out) is False:
            pass 
        else:        
            with open(path_out) as infile:
                for line in infile:
                    window = line.strip().split()
                    window = np.array(list(map(float, window)))  # this is one window, one rep
                    new_rep.append(window)  # current rep often has multiple windows

        # add rep to site list
        if len(new_rep) > 0:
            new_site.append(np.array(new_rep))

    # average across reps
    if len(new_site) > 0:
        print(site, "len(new_site)", len(new_site))
        new_site = np.array(new_site)  # (num_reps, num_windows, window_size)        
        if len(new_site.shape) < 3:
            print("expecting more reps to average, or just write code to deal with this")
            sys.exit()
        new_site = np.mean(new_site, axis=0)    
        outputs.append(new_site)
        X_test.append(X_temp[site])
    else:
        print(site, "len(new_site)", len(new_site), "— missing")
        outputs.append("missing")
        X_test.append("missing")        


# In[13]:


##### replace missing vals with nan
for i in range(len(outputs)):
    if "missing" not in outputs[i]:
        
        # take second year from each time series
        FCH4[i] = FCH4[i][:,timesteps_per_year:(timesteps_per_year*2)]
        FCH4_F_ANNOPTLM[i] = FCH4_F_ANNOPTLM[i][:,timesteps_per_year:(timesteps_per_year*2)]
        X_test[i] = X_test[i][:, timesteps_per_year:(timesteps_per_year*2), :]  

        # identify indices where X OR Y-gap-filled are missing
        missing = ((X_test[i] == -9999).any(dim=-1) | (FCH4_F_ANNOPTLM[i] == -9999) )

        # only evaluate months where X or Y-gap-filled are non-missing
        outputs[i][missing] = np.nan  
        FCH4_F_ANNOPTLM[i][missing] = np.nan 
        
        # replace remaining -9999s with nan
        missing = (FCH4[i] == -9999)
        FCH4[i][missing] = np.nan 
        missing = (FCH4_F_ANNOPTLM[i] == -9999)
        FCH4_F_ANNOPTLM[i][missing] = np.nan 
        # (outputs is never -9999)


# In[14]:


### un-normalize
for i in range(num_sites):
    if "missing" not in outputs[i]:
        outputs[i] = ((outputs[i] * Y_stats[0,1]) + Y_stats[0,0]) / 1000
        FCH4[i] = ((FCH4[i] * M_stats[1]) + M_stats[0]) /1000 
        FCH4_F_ANNOPTLM[i] = ((FCH4_F_ANNOPTLM[i] * G_stats[1]) + G_stats[0]) /1000 


# In[15]:


# fxn to split windows/years into "valid", complete windows, and incomplete windows
def complete_windows(FCH4, FCH4_F_ANNOPTLM, outputs, site):
    num_windows = len(FCH4[site])
    complete_true = []
    missing_true = []
    complete_est = []
    missing_est = []
    for window in range(num_windows):
        nans = torch.sum(np.isnan(FCH4[site][window]))
        if nans == 0:
            complete_true.append(deepcopy(FCH4[site][window]))
            complete_est.append(deepcopy(outputs[site][window]))            
        else:
            missing_true.append(deepcopy(FCH4_F_ANNOPTLM[site][window]))  # years with missing months evaluated against gap-filled fch4
            missing_est.append(deepcopy(outputs[site][window]))            
    #
    return complete_true, missing_true, complete_est, missing_est
    
# for test_ind in range(len(FCH4)):
#     complete_true, missing_true, complete_est, missing_est = complete_windows(FCH4, FCH4_F_ANNOPTLM, outputs, test_ind)
#     print(len(complete_true), len(missing_true), len(complete_est), len(missing_est))


# In[16]:


def yearly(timeseries):
    days_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    tot = 0
    for month in range(12):
        if not np.isnan(timeseries[month]):  # skip nans
            tot += timeseries[month] * days_per_month[month]
    return tot


# In[17]:


### site means and squared errors
true_site_means_valid = []
est_site_means_valid = []
ses_valid = []
est_site_sds_valid = []
annual_variation_valid = []
valid_site_indices = []
#
true_site_means_missing = []
est_site_means_missing = []
ses_missing = []
est_site_sds_missing = []
annual_variation_missing = []
missing_site_indices = []
for i in range(num_sites):
    if "missing" not in outputs[i]:
        complete_true, missing_true, complete_est, missing_est = complete_windows(FCH4, FCH4_F_ANNOPTLM, outputs, i)
        
        # sites with at least one complete window without missing months
        if len(complete_true) > 0:
            valid_site_indices.append(i)
            annual_sums_true = []
            annual_sums_est = []
            squared_errors = []
            for window in range(len(complete_true)):  # (ignoring incomplete windows)
                yearly_true = yearly(complete_true[window])
                yearly_est = yearly(complete_est[window])
                se = (yearly_true-yearly_est)**2
                annual_sums_true.append(yearly_true)
                annual_sums_est.append(yearly_est)
                squared_errors.append(se)
            #
            true_site_means_valid.append( np.mean(annual_sums_true) )
            est_site_means_valid.append( np.mean(annual_sums_est) )
            est_site_sds_valid.append( np.std(annual_sums_est) )
            annual_variation_valid.append( np.std(annual_sums_true) )
            ses_valid.append( np.mean(squared_errors) )
            #
            true_site_means_missing.append( np.mean(annual_sums_true) )  # "complete" sites also included in the "missing" analysis 
            est_site_means_missing.append( np.mean(annual_sums_est) )
            est_site_sds_missing.append( np.std(annual_sums_est) )    
            annual_variation_missing.append( np.std(annual_sums_true) )
            ses_missing.append( np.mean(squared_errors) )

        # sites with no complete windows
        else:  
            missing_site_indices.append(i)
            annual_sums_true = []
            annual_sums_est = []
            squared_errors = []
            for window in range(len(missing_true)):
                nonmissing_months = np.sum(~np.isnan(np.array(missing_true[window])))  # sometimes missing, even though gap filled
                yearly_true = yearly(missing_true[window]) * (12./nonmissing_months)
                yearly_est = yearly(missing_est[window]) * (12./nonmissing_months)  # missing data for outputs was set to match Y-gap-filled
                se = (yearly_true-yearly_est)**2
                annual_sums_true.append( yearly_true )  # scale to 12 months (making up data)
                annual_sums_est.append( yearly_est )
                squared_errors.append(se)                
            #
            true_site_means_missing.append( np.nanmean(annual_sums_true) )  # nan mean because sometimes we train with all gap-filled data
            est_site_means_missing.append( np.nanmean(annual_sums_est) ) 
            est_site_sds_missing.append( np.nanstd(annual_sums_est) )            
            annual_variation_missing.append( np.std(annual_sums_true) )
            ses_missing.append( np.nanmean(squared_errors) )


# In[18]:


### MSE

# sites with complete years
mse_valid = np.sqrt(np.mean(ses_valid))
print(mse_valid)

# including sites with missing months
mse_missing = np.sqrt(np.mean(ses_missing))
print(mse_missing)


# In[19]:


### r2
from sklearn.metrics import r2_score

# complete sites
r2_valid = r2_score(torch.tensor(true_site_means_valid), torch.tensor(est_site_means_valid))
if r2_valid < 0:
    r2_valid = "<0"
print(r2_valid)

# including sites with missing months
r2_missing = r2_score(torch.tensor(true_site_means_missing), torch.tensor(est_site_means_missing))
if r2_missing < 0:
    r2_missing = "<0"
print(r2_missing)


# In[20]:


### correlation

# complete sites
corr_valid = np.corrcoef(torch.tensor(true_site_means_valid), torch.tensor(est_site_means_valid))
print(corr_valid)

# including sites with missing months
corr = np.corrcoef(torch.tensor(true_site_means_missing), torch.tensor(est_site_means_missing))
print(corr)


# In[22]:


### scatter
fig, ax = plt.subplots(figsize=(5, 5)) 
plt.tight_layout()
units=["(g C $m^{-2}$ $year^{-1}$)"]
ax.set_title('Cross-domain model stacking + CNN',fontsize = 15,weight='bold')
ax.set_xlabel('True '+units[0],fontsize = 15)
ax.set_ylabel('Estimated',fontsize = 15)
lim_min=-25
lim_max=300
ax.set_xlim(lim_min,lim_max)
ax.set_ylim(lim_min,lim_max)
x = np.linspace(lim_min,lim_max, 1000)
ax.plot([lim_max*-2,lim_max*2], [lim_max*-2,lim_max*2], color='lightgrey',linestyle='--')
ax.text(-10, 225,
        'RMSE (complete sites) = %0.1f\nRMSE (all sites) = %0.1f' % (mse_valid, mse_missing),
        fontsize = 12, ha='left', va='top')

# all sites (bottom layer)
Y_true=torch.tensor(np.array(true_site_means_missing)[missing_site_indices])  # subset for just the missing sites to avoid plotting points twice
Y_est=torch.tensor(np.array(est_site_means_missing)[missing_site_indices])
Y_sd=torch.tensor(np.array(est_site_sds_missing)[missing_site_indices])
X_eb=torch.tensor(np.array(annual_variation_missing)[missing_site_indices])
# ax.errorbar(Y_true,Y_est, xerr=X_eb, yerr=Y_sd, fmt='^', alpha=0.75, ecolor='black',
#            color="orange", markersize=10)
ax.errorbar(Y_true,Y_est, fmt='^', alpha=0.75, ecolor='black',
           color="orange", markersize=10)

# sites with complete years
Y_true=torch.tensor(true_site_means_valid)
Y_est=torch.tensor(est_site_means_valid)
Y_sd=torch.tensor(est_site_sds_valid)
X_eb=torch.tensor(annual_variation_valid)
# ax.errorbar(Y_true,Y_est, xerr=X_eb, yerr=Y_sd, fmt='o', alpha=0.75, ecolor='black',
#              color="blue", markersize=10)
ax.errorbar(Y_true,Y_est, fmt='o', alpha=0.75, ecolor='black',
             color="blue", markersize=10)

# Create custom legend handles (markers only, no error bars)
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker='o', color='w', label='sites with complete year',
           markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='^', color='w', label='sites with missing months',
           markerfacecolor='orange', markersize=10),
]
plt.legend(handles=legend_handles, loc='upper left', fontsize=12)

plt.show()
fig.savefig(output_path + "ch4.pdf", bbox_inches='tight')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




