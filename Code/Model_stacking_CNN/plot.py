import numpy as np
import matplotlib.pyplot as plt
import math
import os
from io import open
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import random
import sys
from copy import deepcopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
from matplotlib.colors import SymLogNorm
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
from matplotlib.colors import ListedColormap
import rasterio
import config
import utils

### params
start_year, end_year = config.start_year, config.end_year
num_years=end_year-start_year+1
timesteps_per_year=config.timesteps_per_year
timesteps=timesteps_per_year*num_years
days_per_month = config.days_per_month
timesteps_per_year = config.timesteps_per_year
num_windows = config.num_windows
nonmissing_required = config.nonmissing_required

### load gridded data
data0 = torch.load(config.fp_upscale_prep, weights_only=False)
X_grid = data0['X']
Y_tem = torch.tensor(data0['Y'])
Z_grid = data0['Z']
X_vars_grid = data0['X_vars']
Y_vars_tem = data0['Y_vars']
Z_vars_grid = data0['Z_vars']
X_stats_grid = data0['X_stats']
Y_stats_tem = data0['Y_stats']
num_sites = len(Y_tem)

# undo original normalization
X_grid[X_grid == -9999] = np.nan
Y_tem[Y_tem == -9999] = np.nan  # change 9999 to nan
Y_tem = Z_norm_reverse(Y_tem[:,:,0], Y_stats_tem[0,:])  # un-normalize Y
for v in range(len(X_vars_grid)):
    X_grid[:,:,v] = Z_norm_reverse(X_grid[:,:,v], X_stats_grid[v,:])

# shift southern hemisphere back
shift_size = 6  # six months            
for site in range(num_sites):
    lat = Z_grid[site, 0, 1]  # timepoint 0, latitude index=1
    if lat < 0:
        shifted = Y_tem[site, 0:-shift_size].clone()  # new var of shifted data
        Y_tem[site, :] = np.nan  # replace old data with nans
        Y_tem[site, shift_size:] = shifted  # shove in shifted data
        shifted = X_grid[site, 0:-shift_size, :].clone()  # new var of shifted data
        X_grid[site, :, :] = np.nan  # replace old data with nans
        X_grid[site, shift_size:, :] = shifted  # shove in shifted data

######## TEM ##############
def yearly(timeseries):
    days_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    tot = 0
    for month in range(12):
        if not np.isnan(timeseries[month]):  # skip nans
            tot += timeseries[month] * days_per_month[month]
    return tot

def monthly(timeseries):  # convert Tg/m2/month units for Yoummi
    newdata = np.zeros_like(timeseries)  # (12, 12)
    newdata[:] = np.nan
    days_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    for year in range(12):
        for month in range(12):
            if not np.isnan(timeseries[year, month]):  # skip nans
                newdata[year, month] = timeseries[year, month] * days_per_month[month]
    return newdata

### fill global array
long_size,lat_size = 720,360
TEM_estimates = np.empty((lat_size, long_size))  # note the swapped coords
TEM_estimates[:] = np.nan
temp_TEM = np.empty((lat_size, long_size, 12, 12))
temp_TEM[:] = np.nan
area_ind = list(X_vars_grid).index("area")
    
for site in range(len(Z_grid)):
    if site % 10000 == 0:
        print(site)
    long = np.nanmax(Z_grid[site, :, 0])
    lat = np.nanmax(Z_grid[site, :, 1])  
    long_ind,lat_ind = coords2index(long, lat)
    data = Y_tem[site, :]  # (mg / m^2 day)
    data *= 10**6  # square m to square km (mg / km^2 day)    
    data *= 10**-15  # mg to Tg (Tg / km^2 day)    
    data *= X_grid[site, :, area_ind]  # multiply by area (Tg / day)
    data = np.reshape(data, ((int(len(data)/12)), 12))  # (years, months)
    data = data[1:, :]  # cut off first year

    # monthly mean
    temp_TEM[lat_ind, long_ind, :, :] = monthly(data) 

    # annual means
    data = np.nanmean(data, axis=0)   # first get monthly means (across years)— shape (12,)
                                      # this avoids weighting some months unevenly due to missing data 
                                      # I expect more variation seasonally than among years
    data = yearly(data)  # annual sum, across every day of each month (units are now annual)
    TEM_estimates[lat_ind, long_ind] = data  

### global plot
data = deepcopy(TEM_estimates)
data[data <= 0] = np.nan

# define bins
bounds = np.array([0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])

warm_cmap = cm.get_cmap('YlOrRd', len(bounds) - 2)
warm_colors = warm_cmap(np.linspace(0, 1, len(bounds) - 2))

# add a transparent color at the beginning
transparent = np.array([[0.95, 0.95, 0.95, 1]])  # Light gray, visible in both map and colorbar
colors_array = np.vstack([transparent, warm_colors])

# Create new colormap
cmap = ListedColormap(colors_array)
norm = BoundaryNorm(boundaries=bounds, ncolors=len(bounds) - 1)

# Plot
fig = plt.figure(figsize=(11, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

lon = np.linspace(-180, 180, 720)
lat = np.linspace(-90, 90, 360)
heatmap = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                        cmap=cmap, norm=norm)

ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', alpha=0.3)
ax.add_feature(cfeature.OCEAN, facecolor='white', alpha=0.2)
ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)

# Discrete colorbar
cbar = fig.colorbar(heatmap, ax=ax, orientation='horizontal',
                    pad=0.05, fraction=0.035, shrink=0.7, aspect=25,
                    ticks=bounds)
cbar.set_ticklabels(list(map(str, np.round(bounds, 3))))
cbar.set_label('Annual CH₄ flux')

# add text
plt.text(-175, -41.5, "TEM\n" + str(round(TEM_tot, 1)) + r" TgCH$_4$ yr$^{-1}$", fontsize=20,
         # bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3')  # I sort of think it works without the white background
         )

plt.tight_layout()
plt.show()
fig.savefig(config.fp_upscale_out + "/TEM.pdf", bbox_inches='tight')

################ ML ########################

### load ML estimates
ML_estimates = np.empty((lat_size, long_size))  
ML_estimates[:] = np.nan
longlat = np.nanmax(Z_grid, axis=1)
temp_ML = np.empty((lat_size, long_size, 12, 12)) 
temp_ML[:] = np.nan

for site in range(num_sites):
#for site in range(10):  # (for dev purposes, obviously)
    if site % 10000 == 0:
        print(site)

    fp = config.fp_upscale_out + "/estimates_site_" + str(site) + ".npy"
    if os.path.exists(fp) is True:
        #print("\t", site)
        long,lat = longlat[site]
        long_ind,lat_ind = coords2index(long, lat) 
        data0 = np.load(fp)  # (12, 12, 100)
        data0 = np.nanmean(data0, axis=-1)  # mean across (100) independent estimates / training reps

        # monthly mean
        temp_ML[lat_ind, long_ind, :, :] = monthly(data0) 
        
        # annual means
        data0 = np.nanmean(data0, axis=0)   # get MONTHLY means (across years)— shape (12,)
                                             # this avoids weighting some months unevenly due to missing data 
                                             # I expect more variation seasonally than among years
        data0 = yearly(data0)  # annual sum, across every day of each month (units are now annual)
        ML_estimates[lat_ind, long_ind] = data0
        
### plot
data = deepcopy(ML_estimates)
data[data <= 0] = np.nan

# Simulate clustered data near 0
lon = np.linspace(-180, 180, 720)
lat = np.linspace(-90, 90, 360)

warm_cmap = cm.get_cmap('YlOrRd', len(bounds) - 2)
warm_colors = warm_cmap(np.linspace(0, 1, len(bounds) - 2))

# Add a transparent color at the beginning
transparent = np.array([[0.95, 0.95, 0.95, 1]])  # Light gray, visible in both map and colorbar
colors_array = np.vstack([transparent, warm_colors])

# Create new colormap
cmap = ListedColormap(colors_array)
norm = BoundaryNorm(boundaries=bounds, ncolors=len(bounds) - 1)

# Plot
fig = plt.figure(figsize=(11, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

heatmap = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                        cmap=cmap, norm=norm)

ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', alpha=0.3)
ax.add_feature(cfeature.OCEAN, facecolor='white', alpha=0.2)
ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)

# Discrete colorbar
cbar = fig.colorbar(heatmap, ax=ax, orientation='horizontal',
                    pad=0.05, fraction=0.035, shrink=0.7, aspect=25,
                    ticks=bounds)
cbar.set_label('Annual CH₄ flux')
cbar.set_ticklabels(list(map(str, np.round(bounds, 3))))

# add text
plt.text(-175, -41.5, "ML\n" + str(round(ML_tot, 1)) + r" TgCH$_4$ yr$^{-1}$", fontsize=20,
         # bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3')  # I sort of think it works without the white background
         )

plt.tight_layout()
plt.show()
fig.savefig(config.fp_upscale_out + "/ML.pdf", bbox_inches='tight')

### DIFFERENCE plot
data = TEM_estimates - ML_estimates   # Signed difference

lon = np.linspace(-180, 180, 720)
lat = np.linspace(-90, 90, 360)

from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

colors = [
    (0, "#1a9850"),   # green
    (0.5, "white"),   # center
    (1, "#2c7bb6")    # blue
]

cmap = LinearSegmentedColormap.from_list("green_white_blue", colors)

# Normalize with zero as center
vmax = 0.01       # or whatever matches your data distribution
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

fig = plt.figure(figsize=(11, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

heatmap = ax.pcolormesh(
    lon, lat, data,
    transform=ccrs.PlateCarree(),
    cmap=cmap, norm=norm
)

ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', alpha=0.3)
ax.add_feature(cfeature.OCEAN, facecolor='white', alpha=0.2)
ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)

# Colorbar
cbar = fig.colorbar(
    heatmap, ax=ax, orientation='horizontal',
    pad=0.05, fraction=0.035, shrink=0.7, aspect=25
)
cbar.set_label('Annual CH₄ flux difference (TEM - ML)')

# Add ML text annotation
plt.text(
    -175, -41.5,
    "Difference",
    fontsize=20
)

plt.tight_layout()
plt.show()
fig.savefig(config.fp_upscale_out + "/difference.pdf", bbox_inches='tight')


############# hybrid ##############

### load observed data 
data0 = torch.load(config.fp_prep_fluxnet, weights_only=False)
X_obs = data0['X']
Z_obs = data0['Z']
X_vars_obs = data0['X_vars']
Z_vars_obs = data0['Z_vars']
Z_obs = Z_obs[:,:,1:3]  # just lat, long to match formatting of sims
Z_vars_obs = Z_vars_obs[1:3]
X_stats = data0['X_stats']

### swap -9999 for nan
X_obs[X_obs == -9999] = np.nan

# undo original normalization
X_obs[X_obs == -9999] = np.nan
for v in range(len(X_vars_obs)):
    X_obs[:,:,v] = Z_norm_reverse(X_obs[:,:,v], X_stats[v,:])

### record training distribution — ignore wetland type
training_distribution = {}
temp_ind = list(X_vars_obs).index("TA_F")
le_ind = list(X_vars_obs).index("LE_F")
for wt in range(1, 4+1):
    wt_ind = list(X_vars_obs).index("site_class_" + str(wt))
    new_bin = {"temp_min":np.nanmin(X_obs[:,:,temp_ind]),  # Note: min() across all X_obs
               "temp_max":np.nanmax(X_obs[:,:,temp_ind]),
               "le_min":np.nanmin(X_obs[:,:,le_ind]),
               "le_max":np.nanmax(X_obs[:,:,le_ind]),
              }
    training_distribution[wt] = new_bin
#

### fill global array
long_size,lat_size = 720,360
hybrid_estimates = np.empty((lat_size, long_size))  # notice the swapped coords
hybrid_estimates[:] = np.nan
temp_hybrid = np.empty((lat_size, long_size, 12, 12))
temp_hybrid[:] = np.nan
temp_ind = list(X_vars_grid).index("temp")
le_ind = list(X_vars_grid).index("LE")
wt_ind = list(X_vars_grid).index("site_class_1")
count_ML = 0
count_TEM = 0
TEM_sites = []
year_counts = np.zeros((13))
for site in range(len(Z_grid)):
    if site % 10000 == 0:
        print(site)
    long = np.nanmax(Z_grid[site, :, 0])
    lat = np.nanmax(Z_grid[site, :, 1])  
    long_ind,lat_ind = coords2index(long, lat)
    data = X_grid[site, :]  # (mg / m^2 day)
    temp_min = np.nanmin(data[:,temp_ind])
    temp_max = np.nanmax(data[:,temp_ind])
    le_min = np.nanmin(data[:,le_ind])
    le_max = np.nanmax(data[:,le_ind])
    wt = np.nanmax(data[:,wt_ind:wt_ind+4], axis=0)
    if 1 in wt:  # wetland types 1-4
        wt = list(wt).index(1) +1
        if (
            temp_min >= training_distribution[wt]["temp_min"] and 
            temp_max <= training_distribution[wt]["temp_max"] and
            le_min >= training_distribution[wt]["le_min"] and 
            le_max <= training_distribution[wt]["le_max"]
        ):
            hybrid_estimates[lat_ind, long_ind] = ML_estimates[lat_ind, long_ind]
            temp_hybrid[lat_ind, long_ind, :, :] = temp_ML[lat_ind, long_ind, :, :]
            count_ML += 1
        else:
            # print(temp_min, temp_max, le_min, le_max)
            # print(training_distribution[wt])
            # print()
            hybrid_estimates[lat_ind, long_ind] = TEM_estimates[lat_ind, long_ind]
            temp_hybrid[lat_ind, long_ind, :, :] = temp_TEM[lat_ind, long_ind, :, :]
            count_TEM += 1
            TEM_sites.append( np.array([lat, long]) )

        # go in and see which months hit the threshold
        data = data.reshape((13,12,7))
        data = data[:,:, temp_ind]
        extremes = np.any(data.numpy() > training_distribution[wt]["temp_max"], axis=1)
        year_counts += extremes

    else:  # alluvial formation
        hybrid_estimates[lat_ind, long_ind] = TEM_estimates[lat_ind, long_ind]
        temp_hybrid[lat_ind, long_ind, :, :] = temp_TEM[lat_ind, long_ind, :, :]
        count_TEM += 1

### plot

data = deepcopy(hybrid_estimates)
data[data <= 0] = np.nan

# Simulate clustered data near 0
lon = np.linspace(-180, 180, 720)
lat = np.linspace(-90, 90, 360)

warm_cmap = cm.get_cmap('YlOrRd', len(bounds) - 2)
warm_colors = warm_cmap(np.linspace(0, 1, len(bounds) - 2))

# Add a transparent color at the beginning
transparent = np.array([[0.95, 0.95, 0.95, 1]])  # Light gray, visible in both map and colorbar
colors_array = np.vstack([transparent, warm_colors])

# Create new colormap
cmap = ListedColormap(colors_array)
norm = BoundaryNorm(boundaries=bounds, ncolors=len(bounds) - 1)

# Plot
fig = plt.figure(figsize=(11, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# heatmap = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
#                         cmap=cmap)
heatmap = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(),
                        cmap=cmap, norm=norm)

ax.coastlines()
ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', alpha=0.3)
ax.add_feature(cfeature.OCEAN, facecolor='white', alpha=0.2)
ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5)

# Discrete colorbar
cbar = fig.colorbar(heatmap, ax=ax, orientation='horizontal',
                    pad=0.05, fraction=0.035, shrink=0.7, aspect=25,
                    ticks=bounds)
cbar.set_label('Annual CH₄ flux')
cbar.set_ticklabels(list(map(str, np.round(bounds, 3))))

# add text
plt.text(-175, -41.5, "Hybrid\n" + str(round(hyb_tot, 1)) + r" TgCH$_4$ yr$^{-1}$", fontsize=20,
         # bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3')  # I sort of think it works without the white background
         )

# save
plt.tight_layout()
plt.show()
fig.savefig(config.fp_upscale_out + "/hybrid.pdf", bbox_inches='tight')
