import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import torch
import pandas as pd
from scipy.stats import gaussian_kde
import scipy.stats as stats
import geopandas 
import scipy.io
import shapely.geometry
import netCDF4 as nc
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
import rasterio
from copy import deepcopy
import matplotlib.ticker as ticker
import config
import utils

### file paths
model_version = "modelStackCNN"
input_path = config.wd + "/Data/"
output_path = config.wd + "/Out/" + model_version + "/"
path_save_obs = config.wd + "/Out/prep_model.sav"
modis_path = input_path + "/MODIS_fluxnet/"

### load params
start_year, end_year = config.start_year, config.end_year
year_range = end_year-start_year+1
days_per_month = config.days_per_month

##### observed FLUXNET data #####

### read fluxnet data
X_obs = []
counter = 0
with open(input_path + "/fluxnet_emissions_HH.csv") as infile:
    X_vars_obs = np.array(infile.readline().strip().split(","))
    for line in infile:
        counter += 1
        if counter % 100000 == 0:
            print(counter)
        #
        newline = line.strip().split(",")
        if len(newline) != len(X_vars_obs):
            print(newline)
            print("line doesn't have 41 fields; messed up fluxnet preprocessing")
            sys.exit()
        #
        X_obs.append(np.array(newline))
X_obs = np.array(X_obs)
ind = list(X_vars_obs).index("SITE_ID")

keepers = []
keepers.append(0)  # SITE_ID
keepers.append(1)  # lat 
keepers.append(2)  # long
keepers.append(3)  # site_classification
keepers.append(13)  # TIMESTAMP_START
keepers.append(18)  # FCH4. methane flux (raw)
keepers.append(26)  # LE_F. latent heat flux
keepers.append(27)  # FCH4_F. response/target variable, gap-filled
keepers.append(32)  # TA_F. air temp  "Gaps in meteorological variables including air temperature (TA), were filled with ERA-Interim (ERA-I) reanalysis data 
keepers.append(37)  # FCH4_F_ANNOPTLM
X_vars_obs = X_vars_obs[keepers]
X_obs = X_obs[:, keepers]

## neural-net-filled flux
# grab flux column
ind = list(X_vars_obs).index("FCH4_F_ANNOPTLM")
obs = deepcopy(X_obs[:,ind])
obs = obs.astype(float)

### change obs units to match sims
obs = obs * (100. / 72.2)
X_obs[:,ind] = deepcopy(obs)

## gap-filled flux:
# grab flux column
ind = list(X_vars_obs).index("FCH4_F")
obs = deepcopy(X_obs[:,ind])
obs = obs.astype(float)

# convert -9999 to nan
obs[obs == -9999] = np.nan
obs = obs * (100. / 72.2)
X_obs[:,ind] = deepcopy(obs)

## non-gap-filled flux
# grab flux column
ind = list(X_vars_obs).index("FCH4")
obs = deepcopy(X_obs[:,ind])
obs = obs.astype(float)

# convert -9999 to nan
obs[obs == -9999] = np.nan
obs = obs * (100. / 72.2)
X_obs[:,ind] = deepcopy(obs)

### swap out site class for new classes                                                                                                 
wettypes = np.genfromtxt(input_path + '/wetland_classification.txt',dtype='str')
siteid_ind = list(X_vars_obs).index("SITE_ID")
class_ind = list(X_vars_obs).index("SITE_CLASSIFICATION")
for row in range(X_obs.shape[0]):
    if row % 250000 == 0:
        print(row)
    site = X_obs[row,siteid_ind]
    classi = X_obs[row,class_ind]
    if site in wettypes[:,0]:
        ind = list(wettypes[:,0]).index(site)
        new_class = " ".join(wettypes[ind,1:3])
        if new_class == "Forested bog":
            X_obs[row,class_ind] = 1
        elif new_class == "Unforested bog":
            X_obs[row,class_ind] = 2
        elif new_class == "Forested swamp":
            X_obs[row,class_ind] = 3
        elif new_class == "Unforested swamp":
            X_obs[row,class_ind] = 4
        else:
            print("issues")
            sys.exit()
    else:
        print("missing wetland class", row, site, classi)
        sys.exit()

### one-hot vtype
ind = list(X_vars_obs).index('SITE_CLASSIFICATION')
keep_inds = list(range(len(X_vars_obs)))
new_vars = []
keep_inds.remove(ind)  # remove the previous column 
data = np.array(X_obs[:,ind])
cats = list(range(0 +1,4 +1))  # UPDATE
new_static = np.zeros((X_obs.shape[0], len(cats)))
for cat in range(len(cats)):  
    new_col = np.equal(data, str(cats[cat]))
    new_static[:,cat] = np.array(new_col)
    new_vars.append("site_class_"+str(int(cats[cat])))  # original value (intended to be at least)
#
X_obs = X_obs[:, keep_inds]  # update df
X_obs = np.concatenate([X_obs, new_static], axis=1)
X_vars_obs = np.array(X_vars_obs)[keep_inds]  # update vars
X_vars_obs = list(X_vars_obs) + new_vars

### wrap into a time series

# initialize 3d array
num_sites = len(set(list(X_obs[:,0])))
num_vars = X_obs.shape[1]
num_half_hours = year_range * 365 * 48   # years * days per year * half-hours per day
X3d = np.empty((num_sites, num_half_hours, num_vars), dtype = "float32")
X3d[:] = np.nan  # initialize with nans

# wrap
timeindex = list(X_vars_obs).index("TIMESTAMP_START")
siteindex = list(X_vars_obs).index("SITE_ID")
site_dict = {}
site_dict_rev = {}  # use this in later cell
counter = 0
for i in range(len(X_obs)):
    if i % 250000 == 0:
        print(i)
    
    # get site index for current row
    site = X_obs[i,siteindex]
    if site not in site_dict:
        site_dict[site] = deepcopy(counter)
        site_dict_rev[counter] = site
        counter += 1
    #
    siteid = site_dict[site]
    
    # reformat the timestamp
    tp = str(int(float(X_obs[i,timeindex])))
    y,m,d,h,hh = int(tp[0:4]), int(tp[4:6]), int(tp[6:8]), int(tp[8:10]), float(tp[10:12])
    hh = int(hh / 30.0)  # "0" for bottom of the hour; "1" for 30 minutes
    if m == 2 and d == 29:  # skipping feb 29
        pass
    else:
        timepoint = int(hh)  # half hour
        timepoint += (h*2)  # hour (2 half-hours per hour)
        timepoint += (d-1) * 48  # day (48 half-hours per day)
        if m > 1:
            timepoint += np.sum(days_per_month[:(m-1)]) * 48  # month
        timepoint += (y-start_year)*365*48  # year
    
    # shove data in
    X_obs[i, siteindex] = np.nan  # temporarily deleting the site ID `string`; two lines down replace with numeric siteID
    X3d[siteid, timepoint, :] = X_obs[i, :]
    X3d[siteid, timepoint, siteindex] = siteid  # replace site id with numeric site id

X3d = np.delete(X3d, timeindex, axis = 2)  # remove time column
X_vars_obs = np.delete(X_vars_obs, timeindex, axis = 0)
X_obs = np.array(X3d)

### read in MODIS data

# fxn to turn yyyy-mm-dd date into 13yr index
def process_dates(dates):
    timepoints = {}
    for i in range(len(dates)):
        timepoint = dates[i].split("-")
        timepoint = list(map(int,timepoint))
        y = (timepoint[0]-start_year) * 365
        m = np.sum(days_per_month[:timepoint[1]-1])
        d = timepoint[2] - 1
        timepoint = int(y + m + d)  # sometimes float don't know why
        timepoints[timepoint] = i  # key: day of the year where there is data.  
                                   # value: the index of the corresponding image
    return timepoints

def fill_gaps(gappy_data, dates):
    timepoints = process_dates(dates)
    filled_data = []
    for date in range(365*year_range):
        up,down = int(date),int(date)
        counter = 0
        while up not in timepoints and down not in timepoints:
            up += 1
            down -= 1
            counter += 1
            if counter > 365:
                print("bug")
                sys.exit()
        #
        if up in timepoints and down in timepoints:  # tie
            tp1 = timepoints[up]
            tp2 = timepoints[down] 
            v1 = gappy_data[tp1]
            v2 = gappy_data[tp2]
            datum = np.array([v1,v2])  # stacks the two arrays, creating new dimension (0)
            datum = np.nanmean(datum, axis=0)
        elif up in timepoints:
            tp1 = timepoints[up]
            datum = gappy_data[tp1]
        elif down in timepoints:
            tp1 = timepoints[down]
            datum = gappy_data[tp1]
        else:
            print("bug, this shouldn't happen")
            sys.exit()
        #
        filled_data.append( datum )

    #
    filled_data = np.array(filled_data)
    #
    return filled_data

def modland_bits(val):
    bits = np.binary_repr(val, width=32)[::-1]  # reverse order
    return bits[0:2]  # only looking at first two bits

bands = ["sur_refl_b01",
         "sur_refl_b02",
         "sur_refl_b03",
         "sur_refl_b04",
         "sur_refl_b05",
         "sur_refl_b06",
         "sur_refl_b07",
        ]

QC_band = "QA"
ind = list(X_vars_obs).index("SITE_ID")
I_obs = []
for b in range(len(bands)):
    new_band = []
    for site in range(X_obs.shape[0]):        
        site_ID = site_dict_rev[np.nanmax(X_obs[site,:,ind])] 

        # read data
        fp = site_ID + "_" + bands[b]
        with rasterio.open(modis_path +fp + ".tif") as src:
            data = src.read()
    
        # read QC
        fp = site_ID + "_" + QC_band
        with rasterio.open(modis_path + fp + ".tif") as src:
            qc = src.read()
        v_modland_bits = np.vectorize(modland_bits)
        qc = v_modland_bits(qc)
        qc = (qc == '00')  # this means all bands are ideal quality
        
        # filter
        data = np.where(qc, data, np.nan)
        
        # fill gaps
        timestamps = np.load(modis_path + fp + ".npy")
        data = fill_gaps(data, timestamps).astype(float)  # one site, of one band
        
        #
        new_band.append(data)  # combining sites (one band)
    #
    new_band = np.array(new_band)
    I_obs.append(new_band)  # combining bands (all sites)
#
I_obs = np.stack(I_obs, axis = 2)

### replace -9999 with nan in BOTH predictors and ch4 (later, we will convert back; this is for filtering out negative fluxes)
#   this comes AFTER starting a new array that is numeric (and allows nans)
X_obs[np.where(X_obs==-9999)] = np.nan

### monthly average (X)
one_day = 48
one_year = one_day * 365
total_months = 12 * year_range
num_vars = X_obs.shape[2]
newX = np.empty((num_sites, total_months, num_vars))
ind_FCH4 = list(X_vars_obs).index("FCH4")

for year in range(year_range):
    for month in range(12): 
        start = (year*one_year)  # year
        if month > 0:
            start += np.sum(days_per_month[:month]) * one_day  # month
        length_current_month = days_per_month[month] * one_day
        end = start + length_current_month
        month_index = year*12 + month

        for site in range(num_sites):
            data = X_obs[site,start:end,:]
            
            # start by filling the current month with means of each variable— nan, or not.
            newX[site,month_index,:] = np.nanmean(data, axis = 0)
            
            # replace F_CH4 (only) with nan if fewer than 1/7th month of data available (referencing these missing values at test time)        
            timepoints_with_data = 0            
            for hh in range(length_current_month):
                if np.isnan(data[hh,ind_FCH4]):
                    pass
                else:
                    timepoints_with_data += 1
                
            # at least 14% of fluxnet rows from that month non-missing (arbitrary, but kind of matches Gavin's filter)
            perc = float(timepoints_with_data) / float(length_current_month)
            if perc < (1.0/7.0):
                newX[site,month_index,ind_FCH4] = np.nan
#
X_obs = np.array(newX)

### monthly average (I)
one_day = 1
one_year = one_day * 365
total_months = 12 * year_range
num_vars = I_obs.shape[2]
image_width = I_obs.shape[-1]
newX = np.empty((num_sites, total_months, num_vars, image_width, image_width))

for year in range(year_range):
    for month in range(12): 
        start = (year*one_year)  # year
        if month > 0:
            start += np.sum(days_per_month[:month]) * one_day  # month
        length_current_month = days_per_month[month] * one_day
        end = start + length_current_month
        month_index = year*12 + month

        for site in range(num_sites):
            data = I_obs[site, start:end, :, :, :]
            
            # fill the current month with means of each variable— nan, or not.
            means = np.nanmean(data, axis = 0)
            newX[site, month_index, :, :, :] = means

            # make nan where X is nan
            if np.isnan(X_obs[site, month_index, :]).any() is True:
                newX[site, month_index, :, :, :] = np.nan
#
I_obs = np.array(newX)

### offset southern hemisphere by 6 months
shift_size = 6  # six months
lat_ind = list(X_vars_obs).index("LAT")
for site in range(num_sites):
    lat = np.nanmax(X_obs[site,:,lat_ind])
    if lat < 0:
        shifted = deepcopy(X_obs[site,shift_size:,:])  # new var of shifted data
        X_obs[site,:,:] = np.nan  # replace old var with nans
        X_obs[site,0:-shift_size,:] = deepcopy(shifted)  # shove in shifted data
        #
        shifted = deepcopy(I_obs[site,shift_size:,:])  # new var of shifted data
        I_obs[site,:,:,:,:] = np.nan  # replace old var with nans
        I_obs[site,0:-shift_size,:,:,:] = deepcopy(shifted)  # shove in shifted data

### separate out Z: site, lat, long
zinds = [list(X_vars_obs).index('SITE_ID')] + [list(X_vars_obs).index('LAT')] + [list(X_vars_obs).index('LON')]
Z_obs = X_obs[:,:,zinds]
Z_vars_obs = X_vars_obs[zinds]
X_obs = np.delete(X_obs, zinds, axis=2)
X_vars_obs = np.delete(X_vars_obs, zinds)

### separate Y and X
yind = list(X_vars_obs).index("FCH4_F")
Y_obs = X_obs[:,:,yind]
Y_vars_obs = X_vars_obs[yind]
X_obs = np.delete(X_obs, yind, axis=2)
X_vars_obs = np.delete(X_vars_obs, yind)

### z-norm obs INDEPENDENT of sims
Y_obs = np.reshape(Y_obs, (Y_obs.shape[0],Y_obs.shape[1],1))
    
X_stats = np.zeros((X_obs.shape[-1], 2))
Y_stats = np.zeros((1, 2))  
I_stats = np.zeros((I_obs.shape[2], 2))

for v in range(1):
    var = np.nanvar(Y_obs[:,:,v])
    Y_obs[:,:,v], Y_stats[v,0], Y_stats[v,1] = config.Z_norm(Y_obs[:,:,v])

for v in range(len(X_vars_obs)):
    var = np.nanvar(X_obs[:,:,v])
    if var > 0:
        X_obs[:,:,v], X_stats[v,0], X_stats[v,1] = config.Z_norm(X_obs[:,:,v])
    else:
        print("ZERO VARIANCE COLUMN")

for v in range(I_obs.shape[2]):
    var = np.nanvar(I_obs[:,:,v,:,:])
    if var > 0:
        I_obs[:,:,v,:,:], I_stats[v,0], I_stats[v,1] = config.Z_norm(I_obs[:,:,v,:,:])
    else:
        print("ZERO VARIANCE COLUMN")

### replace nan with -9999 (for the GRU)
X_obs = np.nan_to_num(X_obs, nan=-9999)
Y_obs = np.nan_to_num(Y_obs, nan=-9999)
I_obs = np.nan_to_num(I_obs, nan=-9999)

### write
X_obs=torch.tensor(X_obs)
Y_obs=torch.tensor(Y_obs)
Z_obs=torch.tensor(Z_obs)
I_obs=torch.tensor(I_obs)
torch.save({'X': X_obs,
            'Y': Y_obs,
            'Z': Z_obs,
            'I': I_obs,
            'X_stats': X_stats,
            'Y_stats': Y_stats,
            'I_stats': I_stats,
            'X_vars': X_vars_obs,
            'Y_vars': Y_vars_obs,
            'Z_vars': Z_vars_obs,
            }, path_save_obs)
