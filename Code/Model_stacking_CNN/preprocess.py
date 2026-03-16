#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
#from composition_stats import ilr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gc
import rasterio
from copy import deepcopy
import matplotlib.ticker as ticker


# In[2]:


def Z_norm(X):
    X_mean=np.nanmean(X)
    X_std=np.nanstd(np.array(X))
    return (X-X_mean)/X_std, X_mean, X_std


# In[3]:


### file paths
model_version = "161"

input_path = '/scratch.global/chriscs/KGML/Data/Emissions/'
output_path = '/scratch.global/chriscs/KGML/Out/Emissions/v' + model_version + "/"
#input_path = '/Users/chris/TempWorkSpace/KGML/Data/'
#output_path = '/Users/chris/TempWorkSpace/KGML/Out/Emissions/'

path_save_obs = output_path + 'fluxnet_obs_v' + model_version + '.sav'
path_save_sim = output_path + 'fluxnet_sim_v' + model_version + '.sav'

#modis_path = "/Users/chris/TempWorkSpace/MODOS_set6/"
modis_path = input_path + "/MODIS_061625/"


# In[4]:


### params
start_year, end_year = 2006,2018
#start_year, end_year = 2010,2010
year_range = end_year-start_year+1
days_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]

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

def index2coords(long_ind, lat_ind):
    long = float(long_ind) / 2.  # (0,360)
    lat = float(lat_ind) / 2.  # (0,180)
    long -= 180.  # (-180, 180)
    lat -= 90.  # (-90, 90)

    return long, lat


# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


##### observed FLUXNET data #####


# In[6]:


### read fluxnet data
X_obs = []
counter = 0
with open(input_path + "/fluxnet_emissions_HH.csv") as infile:  #3,096,831 lines
    X_vars_obs = np.array(infile.readline().strip().split(","))
    print(X_vars_obs)
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

print(X_vars_obs)
print(len(X_vars_obs))
print(X_obs.shape)
ind = list(X_vars_obs).index("SITE_ID")
print(len(set(list(X_obs[:,ind]))), set(list(X_obs[:,ind])))
print(set(list(X_obs[:,3])))


# In[7]:


### filter some vars   https://fluxnet.org/data/aboutdata/data-variables/

# ['SITE_ID', 'LAT', 'LON', 'SITE_CLASSIFICATION', 'UPLAND_CLASS', 'IGBP', 
# 'KOPPEN', 'MOSS_BROWN', 'MOSS_SPHAGNUM', 'AERENCHYMATOUS', 'ERI_SHRUB', 'TREE', 
# 'DOM_VEG', 'TIMESTAMP_START', 'TIMESTAMP_END', 'NEE', 'H', 'LE', 'FCH4', 'USTAR', 'SW_IN', 'GPP_DT', 
# 'RECO_DT', 'WS', 'NEE_F', 'H_F', 'LE_F', 'FCH4_F', 'SW_IN_F', 'LW_IN_F', 'VPD_F', 
# 'PA_F', 'TA_F', 'P_F', 'WS_F', 'LE_F_ANNOPTLM', 'NEE_F_ANNOPTLM', 
# 'FCH4_F_ANNOPTLM', 'FCH4_F_RANDUNC', 'FCH4_F_ANNOPTLM_UNC', 'FCH4_F_ANNOPTLM_QC']

keepers = []
keepers.append(0)  # SITE_ID
keepers.append(1)  # lat 
keepers.append(2)  # long
keepers.append(3)  # site_classification
#keepers.append(4)  # upland class
#keepers.append(5)  # IGBP

#keepers.append(6)  # KOPPEN
#keepers.append(7)  # MOSS_Brown
#keepers.append(8)  # MOSS_Sphagnum
#keepers.append(9)  # AERENCHYMATOUS
#keepers.append(10)  # ERI_SHRUB
#keepers.append(11)  # TREE

#keepers.append(12)  # DOM_VEG (not sure if we know this info for new grid cells)
keepers.append(13)  # TIMESTAMP_START
#keepers.append(14)  # TIMESTAMP_END
#keepers.append(15)  # NEE. net ecosystem exchange (CO2)- using gap-filled version 
#keepers.append(16)  # H.  Sensible heat flux — using gap-filled version 
#keepers.append(17)  # LE. latent heat flux — using gap-filled version 
keepers.append(18)  # FCH4. methane flux (raw)
#keepers.append(19)  # USTAR. Friction velocity
#keepers.append(20)  # SW_IN. Shortwave radiation, incoming
#keepers.append(21)  # GPP_DT. Gross Primary Productivity (dont know what the "_DT" is)

#keepers.append(22)  # RECO_DT. Ecosystem Respiration (dont know what the "_DT" is)
#keepers.append(23)  # WS. wind speed — using gap-filled
#keepers.append(24)  # NEE_F. net ecosystem exchange (CO2)
#keepers.append(25)  # H_F.  Sensible heat flux
keepers.append(26)  # LE_F. latent heat flux
keepers.append(27)  # FCH4_F. response/target variable, gap-filled
#keepers.append(28)  # SW_IN_F. Shortwave radiation, incoming
#keepers.append(29)  # LW_IN_F. Longwave radiation, incoming
# keepers.append(30)  # VPD_F. Vapor Pressure Deficit

#keepers.append(31)  # PA_F. atmospheric pressure
keepers.append(32)  # TA_F. air temp  "Gaps in meteorological variables including air temperature (TA), were filled with ERA-Interim (ERA-I) reanalysis data (Vuichard and Papale 2015)"
# keepers.append(33)  # P_F. Precipitation
#keepers.append(34)  # WS_F. wind speed
#keepers.append(35)  # LE_F_ANNOPTLM - neural net filled
# keepers.append(36)  # NEE_F_ANNOPTLM

keepers.append(37)  # FCH4_F_ANNOPTLM
# keepers.append(38)  # FCH4_F_RANDUNC
# keepers.append(39)  # FCH4_F_ANNOPTLM_UNC
# keepers.append(40)  # FCH4_F_ANNOPTLM_QC

print(keepers)
X_vars_obs = X_vars_obs[keepers]
X_obs = X_obs[:, keepers]
print(X_vars_obs)
print(len(X_vars_obs))
print(X_obs.shape)


# In[8]:


# # np.save(output_path+"temp1.npy", X_obs)
# # np.save(output_path+"temp2.npy", X_vars_obs)
# X_obs = np.load(output_path+"temp1.npy")
# X_vars_obs = np.load(output_path+"temp2.npy")


# In[9]:


### convert FLUXNET units to match sims


## neural-net-filled flux
# grab flux column
ind = list(X_vars_obs).index("FCH4_F_ANNOPTLM")
obs = deepcopy(X_obs[:,ind])
obs = obs.astype(float)
print(len(obs))

# checking for -9999
print(np.min(obs))
print("-9999 strings", len(obs[obs == "-9999.0"]))
print("-9999 floats", len(obs[obs == -9999.0]))    # these exist for the non-gap filled version; the other is completely gap-filled
print("nans", np.sum(np.isnan(obs)))
   # none!

# # change obs units to match sims
# #  (TEM) mg / day  (FLUXNET) nmol / s    s/day   mol/nmol   g/mol    mg/g    
#                           obs = obs  *  86400 * (10**-9) * 16.04 * (10**3)
# so: 100 mg/day of methane = 72.2 nmol/s
obs = obs * (100. / 72.2)

X_obs[:,ind] = deepcopy(obs)
print(X_obs[:,ind])



## gap-filled flux:
# grab flux column
ind = list(X_vars_obs).index("FCH4_F")
obs = deepcopy(X_obs[:,ind])
obs = obs.astype(float)
print(len(obs))

# checking for -9999
print(np.min(obs))
print("-9999 strings", len(obs[obs == "-9999.0"]))
print("-9999 floats", len(obs[obs == -9999.0]))    # these exist for the non-gap filled version; the other is completely gap-filled
print("nans", np.sum(np.isnan(obs)))

# convert -9999 to nan
obs[obs == -9999] = np.nan
print("-9999 floats", len(obs[obs == -9999.0]))    # these exist for the non-gap filled version; the other is completely gap-filled
print("nans", np.sum(np.isnan(obs)))

obs = obs * (100. / 72.2)
X_obs[:,ind] = deepcopy(obs)
print(X_obs[:,ind])



## non-gap-filled flux
# grab flux column
ind = list(X_vars_obs).index("FCH4")
obs = deepcopy(X_obs[:,ind])
obs = obs.astype(float)
print(len(obs))

# checking for -9999
print(np.min(obs))
print("-9999 strings", len(obs[obs == "-9999.0"]))
print("-9999 floats", len(obs[obs == -9999.0]))    # these exist for the non-gap filled version; the other is completely gap-filled
print("nans", np.sum(np.isnan(obs)))

# convert -9999 to nan
obs[obs == -9999] = np.nan
print("-9999 floats", len(obs[obs == -9999.0]))    # these exist for the non-gap filled version; the other is completely gap-filled
print("nans", np.sum(np.isnan(obs)))

obs = obs * (100. / 72.2)
X_obs[:,ind] = deepcopy(obs)
print(X_obs[:,ind])


# In[10]:


### swap out site class for new classes                                                                                                 
wettypes = np.genfromtxt(input_path + '/wetland_classification.txt',dtype='str')
print(wettypes)
print(wettypes.shape)
print(X_vars_obs)
siteid_ind = list(X_vars_obs).index("SITE_ID")
class_ind = list(X_vars_obs).index("SITE_CLASSIFICATION")
print(siteid_ind, class_ind)
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

print(set(list(X_obs[:,class_ind])))
print(list(X_obs[:,class_ind]).count("1"))
print(list(X_obs[:,class_ind]).count("2"))
print(list(X_obs[:,class_ind]).count("3"))
print(list(X_obs[:,class_ind]).count("4"))


# In[11]:


### one-hot vtype
ind = list(X_vars_obs).index('SITE_CLASSIFICATION')
keep_inds = list(range(len(X_vars_obs)))
new_vars = []
keep_inds.remove(ind)  # remove the previous column 
data = np.array(X_obs[:,ind])
print(data.shape)
#cats = list(set(list(data)))
cats = list(range(0 +1,4 +1))  # UPDATE
print(len(cats), cats)
new_static = np.zeros((X_obs.shape[0], len(cats)))
for cat in range(len(cats)):  
    new_col = np.equal(data, str(cats[cat]))
    print(cats[cat], np.sum(new_col))
    new_static[:,cat] = np.array(new_col)
    #new_vars.append(v+"_"+str(cat))  # this way re-codes the vtype to a new index
    new_vars.append("site_class_"+str(int(cats[cat])))  # original value (intended to be at least)

X_obs = X_obs[:, keep_inds]  # update df
X_obs = np.concatenate([X_obs, new_static], axis=1)

X_vars_obs = np.array(X_vars_obs)[keep_inds]  # update vars
X_vars_obs = list(X_vars_obs) + new_vars

print(X_obs.shape)
print(len(X_vars_obs))
print(X_vars_obs)


# In[12]:


### wrap into a time series

# initialize 3d array
num_sites = len(set(list(X_obs[:,0])))
num_vars = X_obs.shape[1]
num_half_hours = year_range * 365 * 48   # years * days per year * half-hours per day
X3d = np.empty((num_sites, num_half_hours, num_vars), dtype = "float32")
X3d[:] = np.nan  # initialize with nans
print(X3d.shape)

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
    #print(tp)
    y,m,d,h,hh = int(tp[0:4]), int(tp[4:6]), int(tp[6:8]), int(tp[8:10]), float(tp[10:12])
    hh = int(hh / 30.0)  # "0" for bottom of the hour; "1" for 30 minutes
    #print(y,m,d,h,hh)
    if m == 2 and d == 29:  # skipping feb 29
        pass
    else:
        timepoint = int(hh)  # half hour
        timepoint += (h*2)  # hour (2 half-hours per hour)
        timepoint += (d-1) * 48  # day (48 half-hours per day)
        if m > 1:
            timepoint += np.sum(days_per_month[:(m-1)]) * 48  # month
        timepoint += (y-start_year)*365*48  # year
    #print(timepoint)
    
    # shove data in
    X_obs[i, siteindex] = np.nan  # temporarily deleting the site ID `string`; two lines down replace with numeric siteID
    X3d[siteid, timepoint, :] = X_obs[i, :]
    X3d[siteid, timepoint, siteindex] = siteid  # replace site id with numeric site id

X3d = np.delete(X3d, timeindex, axis = 2)  # remove time column
X_vars_obs = np.delete(X_vars_obs, timeindex, axis = 0)
X_obs = np.array(X3d)
print(X_obs.shape)


# In[13]:


# # np.save(output_path+"temp1.npy", X_obs)
# # np.save(output_path+"temp2.npy", X_vars_obs)
# X_obs = np.load(output_path+"temp1.npy")
# X_vars_obs = np.load(output_path+"temp2.npy")


# In[ ]:


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
    print("\t", bands[b])
    new_band = []
    for site in range(X_obs.shape[0]):        
        site_ID = site_dict_rev[np.nanmax(X_obs[site,:,ind])] 
        print(site, site_ID)

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
        #print(site_ID, year, bands[b], data.shape, len(timestamps))
        data = fill_gaps(data, timestamps).astype(float)  # one site, of one band
        
        #
        new_band.append(data)  # combining sites (one band)
    #
    new_band = np.array(new_band)
    I_obs.append(new_band)  # combining bands (all sites)
#
I_obs = np.stack(I_obs, axis = 2)
print(I_obs.shape)  # (sites, timesteps, bands, width, height)


# In[20]:


### visualize one image

for i in range(0,100):
    print(i)
    fp = site_ID + "_sur_refl_b01"
    with rasterio.open(modis_path +fp + ".tif") as src:
        r = src.read()[i]
    fp = site_ID + "_sur_refl_b04"
    with rasterio.open(modis_path +fp + ".tif") as src:
        g = src.read()[i]
    fp = site_ID + "_sur_refl_b03"
    with rasterio.open(modis_path +fp + ".tif") as src:
        b = src.read()[i]
    #
    rgb = np.stack((r,g,b), axis=-1)
    rgb = rgb / rgb.max()
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()


# In[27]:


# # # np.save(output_path+"temp5.npy", I_obs)
# I_obs = np.load(output_path+"temp5.npy")
# print(I_obs.shape)


# In[28]:


### replace -9999 with nan in BOTH predictors and ch4 (later, we will convert back; this is for filtering out negative fluxes)
#   this comes AFTER starting a new array that is numeric (and allows nans)

print(np.sum(np.isnan(X_obs)))
print(len(np.where(X_obs == -9999)[0]))
X_obs[np.where(X_obs==-9999)] = np.nan
print(np.sum(np.isnan(X_obs)))
print(len(np.where(X_obs == -9999)[0]))

# (I_obs is already nan)


# In[29]:


### monthly average (X)
print(X_obs.shape)
print(X_vars_obs)
one_day = 48
one_year = one_day * 365
total_months = 12 * year_range
num_vars = X_obs.shape[2]
newX = np.empty((num_sites, total_months, num_vars))
ind_FCH4 = list(X_vars_obs).index("FCH4")

for year in range(year_range):
    print(year)
    for month in range(12): 
        start = (year*one_year)  # year
        if month > 0:
            start += np.sum(days_per_month[:month]) * one_day  # month
        length_current_month = days_per_month[month] * one_day
        end = start + length_current_month
        month_index = year*12 + month

        for site in range(num_sites):
            data = X_obs[site,start:end,:]
            #print(data.shape)  # (1488, 11)
            
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

X_obs = np.array(newX)
print(X_obs.shape)


# In[30]:


### monthly average (I)
print(I_obs.shape)
one_day = 1
one_year = one_day * 365
total_months = 12 * year_range
num_vars = I_obs.shape[2]
image_width = I_obs.shape[-1]
newX = np.empty((num_sites, total_months, num_vars, image_width, image_width))

for year in range(year_range):
    print(year)
    for month in range(12): 
        start = (year*one_year)  # year
        if month > 0:
            start += np.sum(days_per_month[:month]) * one_day  # month
        length_current_month = days_per_month[month] * one_day
        end = start + length_current_month
        month_index = year*12 + month

        for site in range(num_sites):
            data = I_obs[site, start:end, :, :, :]
            # print(data.shape)  # (31, 2, 10, 10)
            
            # fill the current month with means of each variable— nan, or not.
            means = np.nanmean(data, axis = 0)
            newX[site, month_index, :, :, :] = means

            # make nan where X is nan
            if np.isnan(X_obs[site, month_index, :]).any() is True:
                newX[site, month_index, :, :, :] = np.nan

I_obs = np.array(newX)
print(I_obs.shape)


# In[31]:


### offset southern hemisphere by 6 months
print(X_obs.shape)
print(I_obs.shape)
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


# In[32]:


### augment modis images
print(I_obs.shape)


# In[33]:


### separate out Z: site, lat, long
zinds = [list(X_vars_obs).index('SITE_ID')] + [list(X_vars_obs).index('LAT')] + [list(X_vars_obs).index('LON')]
Z_obs = X_obs[:,:,zinds]
Z_vars_obs = X_vars_obs[zinds]
print(Z_vars_obs)
print(Z_obs.shape)

X_obs = np.delete(X_obs, zinds, axis=2)
X_vars_obs = np.delete(X_vars_obs, zinds)
print(X_vars_obs)
print(X_vars_obs.shape)
print(X_obs.shape)


# In[34]:


### separate Y and X
yind = list(X_vars_obs).index("FCH4_F")
Y_obs = X_obs[:,:,yind]
Y_vars_obs = X_vars_obs[yind]
print(Y_vars_obs, Y_obs.shape)

X_obs = np.delete(X_obs, yind, axis=2)
X_vars_obs = np.delete(X_vars_obs, yind)

# # remove the extra flux column too
# yind = list(X_vars_obs).index("FCH4")
# X_obs = np.delete(X_obs, yind, axis=2)
# X_vars_obs = np.delete(X_vars_obs, yind)

print(X_vars_obs)
print(X_vars_obs.shape)
print(X_obs.shape)


# In[35]:


# ### rearrange vars to match sims

# # sim vars
# # ['LE' 'temp' 'site_class_1' 'site_class_2' 'site_class_3' 'site_class_4']

# #     0       1         2               3              4              5              
# #
# # obs vars
# # ['FCH4' 'LE_F' 'TA_F' 'FCH4_F_ANNOPTLM' 'site_class_1' 'site_class_2' 'site_class_3' 'site_class_4']

# #     0     1      2            3               4              5            6                7

# inds = [1, 2, 4, 5, 6, 7, 0, 3]  # indices vector
# X_obs = X_obs[:, :, inds]
# X_vars_obs = X_vars_obs[inds]
# print(X_obs.shape)
# print(len(X_vars_obs))
# print(X_vars_obs)


# In[36]:


# np.save(output_path+"temp3.npy", X_obs)
# np.save(output_path+"temp4.npy", X_vars_obs)
# X_obs = np.load(output_path+"temp3.npy")
# X_vars_obs = np.load(output_path+"temp4.npy")


# In[37]:


# ### time series plot
# ind_precip = list(X_vars_obs).index("P_F")
# ind_temp = list(X_vars_obs).index("TA_F")

# for site in range(num_sites):
#     fig, ax = plt.subplots(1, 1,figsize=(30, 5*1))
#     plt.rcParams.update({'font.size': 22})
#     # units="(TEM units)"
#     timearr=np.arange(X_obs.shape[1])    

#     # # offset southern hemisphere
#     # lat = np.nanmax(Z_obs[site,:,1])
#     # if lat > 0:     
#     #     offset = 0
#     # else:
#     #     offset = 6
#     offset = 0

#     data = deepcopy(X_obs[site, timearr, ind_precip])
#     data,_,_ = Z_norm(data)
#     ax.plot(timearr-offset, data, "o-", label="precip", color='blue')

#     data = deepcopy(X_obs[site, timearr, ind_temp])
#     data,_,_ = Z_norm(data)
#     ax.plot(timearr-offset, data, "o-", label="air_temp", color='red')

#     data = deepcopy(Y_obs[site, timearr])
#     data,_,_ = Z_norm(data)
#     ax.plot(timearr-offset, data, "o-", label="ch4 flux", color='green')
    
#     ax.set_xlim((timearr[0]-offset,timearr[-1]-offset))
#     ax.set_xlabel("time (months)")
#     # ax.set_ylabel(units)
#     ax.set_title(str(round(np.nanmax(Z_obs[site,:,1]),3)) + ", " + str(round(np.nanmax(Z_obs[site,:,2]),3)))
#     ax.grid()
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
#     ax.legend()
#     plt.show()

# #fig.savefig("/Users/chris/TempWorkSpace/ch4_" + site + ".pdf", bbox_inches='tight')


# In[38]:


##### add TEM output as predictor #####


# In[39]:


# # load simulated data
# fp="/scratch.global/chriscs/KGML/Out/Emissions/fluxnet_sim_v1.sav"
# data0 = torch.load(fp, weights_only=False)
# X_sim = torch.tensor(data0['X'])
# Y_sim = torch.tensor(data0['Y'])
# Y_stats_sim = torch.tensor(data0['Y_stats'])
# X_stats_sim = torch.tensor(data0['X_stats'])
# Z_sim = data0['Z']
# X_vars_sim = data0['X_vars']
# Z_vars_sim = data0['Z_vars']
# # Y_stats = data0['Y_stats']
# print(Y_sim.shape, flush=True)
# print(Z_sim.shape, flush=True)
# print(X_vars_sim, flush=True)
# print(Z_vars_sim, flush=True)

# # convert -9999 to nan
# print(torch.sum(np.isnan(Y_sim)))
# print(len(np.where(Y_sim == -9999)[0]))
# Y_sim[np.where(Y_sim==-9999)] = np.nan
# print(torch.sum(np.isnan(Y_sim)))
# print(len(np.where(Y_sim == -9999)[0]))


# In[40]:


# ### align by lat long
# print(X_obs.shape)
# num_sites = Y_obs.shape[0]
# num_timepoints = Y_obs.shape[1]
# hits = []
# locs = []
# new_X = []
# new_Y = []
# new_Z = []
# new_TEM = []
# for site in range(num_sites):
#     print(site)
#     lat_ind = list(Z_vars_obs).index("LAT")
#     long_ind = list(Z_vars_obs).index("LON")
#     lat = np.nanmax(Z_obs[site,:,lat_ind])
#     long = np.nanmax(Z_obs[site,:,long_ind])
#     # print(lat, long)
#     lat = np.floor(lat*2)/2
#     long = np.floor(long*2)/2
#     # print(lat, long)

#     # # print(np.where((Z_sim==lat) & (Z_sim==long)))
#     # print(np.where((Z_sim==lat) & (Z_sim==long)))
#     lat_ind = list(Z_vars_sim).index("lat")
#     long_ind = list(Z_vars_sim).index("long")        
#     hit = np.where((Z_sim[:,0,lat_ind] == lat) & (Z_sim[:,0,long_ind] == long))[0]
#     if len(hit) == 0:  # opening up to considering nearby grid cells.
#         done = False
#         radius = 0
#         while done == False:
#             radius += 0.5
#             hit = np.where((Z_sim[:,0,lat_ind] >= lat-radius) & 
#                            (Z_sim[:,0,lat_ind] <= lat+radius) &
#                            (Z_sim[:,0,long_ind] >= long-radius) &
#                            (Z_sim[:,0,long_ind] <= long+radius))[0]
#             if len(hit) > 0:
#                 done = True
#                 # for h in hit:
#                 #     coords = Z_sim[h,0,:].numpy()
#                 #     print(coords[1], coords[0])

#     #
#     # print(Z_sim[hit[0],0,:].numpy())
#     hits.append(hit[0])  # in cases with multiple hits just taking the first (random) one
#     #obs_data = Y_obs[site, :, 0]
#     sim_data = Y_sim[hit[0], :]
#     # sim_data = np.expand_dims(sim_data, axis=1)
#     new_TEM.append(np.array(sim_data))
#     # new_data = np.concatenate([X_obs[site],sim_data], axis=1)
#     new_data = np.array(X_obs[site])
#     new_X.append(new_data)
#     new_data = np.array(Y_obs[site])        
#     new_Y.append(new_data)
#     new_data = np.array(Z_obs[site])        
#     new_Z.append(new_data)
#     #print(obs_data.shape, sim_data.shape, np.array([obs_data, sim_data]).shape)
#     #Y.append(np.array([obs_data, sim_data]))
#     locs.append([lat, long])


    
#     #print()
        

# #
# print(len(hits), len(set(hits)))  # I guess repeated hits means flux towers fall inside the same grid cell?
# X_obs = np.array(new_X)
# Y_obs = np.array(new_Y)
# Z_obs = np.array(new_Z)
# TEM_data = np.array(new_TEM)
# print(X_obs.shape)
# print(Y_obs.shape)
# print(Z_obs.shape)
# print(TEM_data.shape)
# # X_vars_obs = np.append(X_vars_obs, "tem_flux")
# print(X_vars_obs)


# In[41]:


# ### time series plot
# for site in range(len(X_obs)):
#     fig, ax = plt.subplots(1, 1,figsize=(30, 5*1))
#     plt.rcParams.update({'font.size': 22})
#     # units="(TEM units)"
#     timearr=np.arange(X_obs.shape[1])    
#     offset = 0

#     data = deepcopy(Y_obs[site, timearr])
#     # data,_,_ = Z_norm(data)
#     ax.plot(timearr-offset, data, "o-", label="observed", color='blue')

#     data = deepcopy(TEM_data[site, timearr, 0])
#     # data,_,_ = Z_norm(data)
#     data = (data * Y_stats_sim[0,1].numpy()) + Y_stats_sim[0,0].numpy()
#     ax.plot(timearr-offset, data, "o-", label="TEM", color='red')
    
#     ax.set_xlim((timearr[0]-offset,timearr[-1]-offset))
#     ax.set_xlabel("time (months)")
#     # ax.set_ylabel(units)
#     ax.set_title("site " + str(site) + ": " + str(round(np.nanmax(Z_obs[site,:,1]),3)) + ", " + str(round(np.nanmax(Z_obs[site,:,2]),3)))
#     ax.grid()
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(12))
#     ax.legend()
#     plt.show()

# #fig.savefig("/Users/chris/TempWorkSpace/ch4_" + site + ".pdf", bbox_inches='tight')


# In[42]:


### z-norm obs INDEPENDENT of sims
Y_obs = np.reshape(Y_obs, (Y_obs.shape[0],Y_obs.shape[1],1))
    
X_stats = np.zeros((X_obs.shape[-1], 2))
Y_stats = np.zeros((1, 2))  
I_stats = np.zeros((I_obs.shape[2], 2))

for v in range(1):
    var = np.nanvar(Y_obs[:,:,v])
    print(v, var)
    Y_obs[:,:,v], Y_stats[v,0], Y_stats[v,1] = Z_norm(Y_obs[:,:,v])

print()
for v in range(len(X_vars_obs)):
    var = np.nanvar(X_obs[:,:,v])
    print(v, var)
    if var > 0:
        X_obs[:,:,v], X_stats[v,0], X_stats[v,1] = Z_norm(X_obs[:,:,v])
    else:
        print("ZERO VARIANCE COLUMN")

print()
for v in range(I_obs.shape[2]):
    var = np.nanvar(I_obs[:,:,v,:,:])
    print(v, var)
    if var > 0:
        I_obs[:,:,v,:,:], I_stats[v,0], I_stats[v,1] = Z_norm(I_obs[:,:,v,:,:])
    else:
        print("ZERO VARIANCE COLUMN")


# In[43]:


### replace nan with -9999 (for the GRU)
X_obs = np.nan_to_num(X_obs, nan=-9999)
Y_obs = np.nan_to_num(Y_obs, nan=-9999)
I_obs = np.nan_to_num(I_obs, nan=-9999)


# In[44]:


### save the data

print(X_obs.shape)
print(Y_obs.shape)
print(Z_obs.shape)
print(I_obs.shape)

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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




