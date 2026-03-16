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
import random
import config
import utils

### file paths
input_path = config.wd + "/Data/"
output_path = confif_wd + "/Out/"
path_save_sim = output_path + '/fluxnet_sim.sav'

def load_predictor(fpath, header, target_year, ch4=False):
    print("loading predictor", fpath)
    counter = 0
    var = np.full((720, 360, 12*31), np.nan)
    long_ind = header.index("LONG")  # index for "LONG" in the list of variables
    lat_ind = header.index("LAT")
    year_ind = header.index("YEAR")
    with open(fpath) as infile:
        for line in infile:
            counter += 1
            newline = line.strip().split(",")
            year = int(newline[year_ind])                  
            if year == target_year:
                long = float(newline[long_ind])  # the longitude coordinate of the grid cell
                lat = float(newline[lat_ind])
                long_cell, lat_cell = coords2index(long, lat)  # convert the lat long to array indices starting from 0 (instead of negatives)
                data = newline[len(header):len(newline)-1]  
                data = np.array(data).astype(np.float32)
                data[data == -999999.0] = np.nan  # Ch4
                data[data == -999999.9] = np.nan  # LAI (.9 versus .0)
                data[data == -99999.0] = np.nan  # FRIN
                duplicate = False  # default (see below)
                if "MONTH" in header:
                    month_ind = header.index("MONTH")
                    month = int(newline[month_ind])                      
                
                # process ch4 data uniquely
                if ch4 == True:
                    data *= -1  # flip sign (TEM emissions are negative -Youmi)
                    
                    # check for duplicates (doing this row-wise is intentional; because <31-day months will have nan for 31st day. The missing vals are rare, never whole row.)
                    start = ((month-1)*31)  # range of days covered by current row
                    end = ((month-1)*31) + len(data)
                    previous = var[long_cell, lat_cell, start:end]  # checking if data already exist for those time points in our current array
                    num_nans = np.sum(np.isnan(previous))
                    if num_nans < len(data):  # if not all nans in our current array, then we already have data for this month
                        duplicate = True  # if all nans in our current array, then we don't already have data for this month
                #
                if len(data) == 12:  # monthly resolution
                    for month in range(len(data)):
                        timepoint = (month*31) + 0  # convert month, year to timepoint index
                        var[long_cell, lat_cell, timepoint:(timepoint+31)] = data[month]  # shove in, repeating for all days in month
                else:  # daily resolution
                    for day in range(len(data)):    
                        timepoint = ((month-1)*31) + day  # convert day, month, year to timepoint index
                        if duplicate is False:
                            var[long_cell, lat_cell, timepoint] = data[day]  # shove in
                        else:  # average with duplicate site-month lines (-Licheng)
                            var[long_cell, lat_cell, timepoint] = (previous[day] + data[day])/2  # this operation should (intentionally) produce nan for averages involving nans
    
    #
    var = np.expand_dims(var, axis=-1)            
    return var

### divide "emissions" values by "frin"
# currently values are weighted by wetland area, I want to divide by wetland area to get the output from wetlands
def divide_frin(X, FRIN):
    ch4_only = np.array(X).squeeze()
    frin = np.array(FRIN).squeeze()

    # save indices where ch4 is missing
    missing = np.where(np.isnan(ch4_only))
    
    # divide (note: FRIN is often nan, outputs nan)
    frin /= 10000  # "The sg_frin is the 10000*(fraction of wetland area), so you may want to convert them to actual fraction first before dividing. The unit should be mg m^-2 day^-1 in this way. You can also exclude any numbers out of 2000 after conversion"
    ch4_only /= frin
    
    # convert -99999 to 0: FRIN is stupidly -99999.0 where fraction inundated is 0
    frin = np.nan_to_num(frin, nan=0)
    ch4_only = np.nan_to_num(ch4_only, nan=0)  # emissions always 0 where FRIN == -99999 (I checked)

    # where ch4 was genuinely missing, change to missing.
    ch4_only[missing] = np.nan
                                                                        
    # filter ch4 fluxes values greater than 2000, says lLicheng
    big_ch4 = np.where(ch4_only > 2000)
    ch4_only[big_ch4] = np.nan
    
    # replace X with filtered data
    X[:,:,:,0] = ch4_only
    del ch4_only, frin, big_ch4, missing
    gc.collect()
    return X

def load_siteclass(header):
    print("loading site classification")
    fp = input_path + "/wet1x1.tem"
    data = np.genfromtxt(input_path + "/wet1x1.tem", delimiter=',')
    counter = 0
    var = np.full((720, 360, 12*31, 1), np.nan)
    long_ind = header.index("LONG")
    lat_ind = header.index("LAT")
    class_ind = header.index("CLASS")
    for row in range(data.shape[0]):
        long = float(data[row, long_ind]) 
        lat = float(data[row, lat_ind]) 
        long_cell, lat_cell = coords2index(long, lat)  # convert the lat long here to array indices starting from 0
        site_class = data[row, class_ind]

        # deal with -99999s — commonly—but not consistently—over water — currently filtering these because don't know what else to do
        if site_class == -99999:
            site_class = np.nan
    
        # deal with (+)99999s — appears to be land a lot of the time (and much more common than any other value)
        elif site_class == 99999:
            site_class = np.nan
    
        # include alluvials because need to apply TEM or a TEM-surrogate for upscaling

        # repeat the data for all time points
        site_class = np.expand_dims(site_class, axis=0)
        site_class = np.repeat(site_class, 12*31, axis=0)
    
        # stuff in
        var[long_cell, lat_cell, :, 0] = site_class
        
    #
    return var

def timeseries(data):
    print("converting to timeseries")
    var = []
    class_ind = X_vars.index("class")
    ch4_ind = X_vars.index("ch4")
    for long in range(720):
        for lat in range(360):
            datum = data[long, lat, :, :]
            if not np.any(np.isnan(datum[:,class_ind])):  # has a wetland classification wet1x1.tem
                long_real, lat_real = index2coords(long,lat)  # convert back to real lat long
                longlat = np.array([long_real, lat_real])  # repeat the data for all time points
                longlat = np.expand_dims(longlat, axis=0)
                longlat = np.repeat(longlat, 12*31, axis=0)        
                datum = np.concatenate([datum, longlat], axis=1)  # add long lat to data
                var.append(np.array(datum))
    #
    var = np.array(var)
    return var

def load_LE(fp): # era 5
    print("loading LE", fp)
    data = np.full((720, 360, 1, 31), np.nan)
    data[:] = np.nan
    nc_file = nc.Dataset(fp)  

    LE = nc_file['slhf']  # valid_time(1), latitude(1801), longitude(3600) 
    LE = np.array(LE).squeeze()#.astype(float)  (1801, 3600)
    LE *= -1  # flip sign
    seconds = 60*60*24  # seconds per DAY ("monthly means of daily means"—https://confluence-test-dc.ecmwf.int/display/CKB/ERA5%3A%2Bdata%2Bdocumentation?)
    LE /= seconds

    # convert to 0.5 degrees
    ratio = int(LE.shape[1] / 720)
    #for long_ind in range(720):

    # dealing with the first half first: -180 to 0 degrees 
    for long_ind in range(0,360):  
        long_start = (long_ind*ratio)   
        long_end = long_start + ratio

        # adjust longitude
        long_start,long_end = long_start + int(LE.shape[1]/2), long_end + int(LE.shape[1]/2)
        
        for lat_ind in range(360):
            lat_start = (lat_ind*ratio) 
            lat_end = lat_start + ratio

            # flip latitude
            lat_start,lat_end = LE.shape[0]-lat_end, LE.shape[0]-lat_start
                            
            # shove in
            square = LE[lat_start:lat_end, long_start:long_end].copy()
            data[long_ind, lat_ind, :, :] = np.nanmean(square)  # repeat for every day in month

    # 0 to 180 degrees
    for long_ind in range(360, 720): 
        long_start = (long_ind*ratio)   
        long_end = long_start + ratio

        # adjust longitude
        long_start,long_end = long_start - int(LE.shape[1]/2), long_end - int(LE.shape[1]/2)

        for lat_ind in range(360):
            lat_start = (lat_ind*ratio) 
            lat_end = lat_start + ratio

            # flip latitude
            lat_start,lat_end = LE.shape[0]-lat_end, LE.shape[0]-lat_start
                            
            # shove in
            square = LE[lat_start:lat_end, long_start:long_end].copy()
            data[long_ind, lat_ind, :, :] = np.nanmean(square)  # repeat for every day in month
    
    #
    return data


### read data
X_vars = ["ch4", "LE", "temp", "class", "long", "lat"] 
X = []
for year in range(config.start_year, config.end_year+1):
    print(year, flush=True)
    variables = []

    # ch4
    fpath = input_path + "ch4emi.day"
    header = ["LONG", "LAT", "VAR_NAME", "DONTKNOW", "DONTKNOW", "DONTKNOW", "DONTKNOW", "AREA", "YEAR", "MONTH", "SUM", "MAX", "MEAN", "MIN"]
    #         -180.0,  65.0,  CH4EMI ,      3,            3,        55.00,      0,        1294,   1978,    1,      0.0,   0.0,   0.00,   0.0
    var = load_predictor(fpath, header, year, ch4=True)

    # frin
    fpath = input_path + "sg_frin.tem"    
    header = ["LONG", "LAT", "VAR_NAME", "DONTKNOW", "YEAR", "SUM", "min", "mean", "max"]
    if year <= 2012:
        FRIN = load_predictor(fpath, header, year)
    else:  # use 2012 data for later years
        FRIN = load_predictor(fpath, header, 2012)
    #
    var = divide_frin(var, FRIN)  # divide ch4 by frin
    variables.append(var)  # add the updated-ch4 to list

    # LE (era5)
    var = []
    for month in range(12):
        fpath = input_path + 'LE_land/era5_land_slhf_monthly_' + str(year) + "_" + str(month+1).zfill(2) + '.nc'
        var.append(load_LE(fpath))
    var = np.concatenate(var, axis = 2)
    var = np.reshape(var, (720, 360, 372, 1))
    variables.append(var)
    
    # temperature
    fpath = input_path + "ecmwf_TAIR_1979-2018.tem"
    header = ["LONG", "LAT", "VAR_NAME", "DONTKNOW", "YEAR", "MONTH", "SUM", "min", "mean", "max"]
    T = load_predictor(fpath, header, year)
    variables.append(T)
        
    # site classification  — necessary for current code to work
    header = ["LONG", "LAT", "WET", "DONTKNOW", "CLASS", "CONTINENT"]
    var = load_siteclass(header)
    variables.append(var)
    
    # reshape into time series for each site/location — simultaneously filter for wetlands, add on lat, long
    variables = np.concatenate(variables, axis=-1)
    variables = timeseries(variables)
    print(variables.shape)
    X.append(variables)
#
del var, T, FRIN
gc.collect()
X_backup = list(X)
X = np.concatenate(X, axis=1)

### filter sites with all misssing data
ch4_ind = list(X_vars).index("ch4")
arr = []
for site in range(X.shape[0]):
    if site % 1000 == 0:
        print(site)
    if not np.all(np.isnan(X[site,:,ch4_ind])):  # not all missing ch4 data (e.g. ocean, non-wetland sites)    
        arr.append(np.array(X[site,:,:]))
        
#
X = np.array(arr)
del arr
gc.collect()

### one-hot site class (doing this separate from above to save memory)
ind = list(X_vars).index('class')
keep_inds = list(range(len(X_vars)))
new_vars = []
keep_inds.remove(ind)  # remove the previous column 
data = np.array(X[:,:,ind])
cats = [1,2,3,4]
new_static = np.zeros((X.shape[0], X.shape[1], len(cats)))
for cat in range(len(cats)):
    new_col = np.equal(data, cats[cat])
    new_static[:,:,cat] = np.array(new_col)
    new_vars.append("site_class_"+str(int(cats[cat])))  # original value (intended to be at least)

X = X[:, :, keep_inds]  
X = np.concatenate([X, new_static], axis=2)

X_vars = np.array(X_vars)[keep_inds]  
X_vars = list(X_vars) + new_vars

### filter time series for "real" days (instead of assuming 31 days per month)
keepers = []
for month in range(12):
    for day in range(31):
        if day < config.days_per_month[month]:
            keepers.append(True)
        else:
            keepers.append(False)

keepers *= config.year_range  # repeat for all years
X = X[:, keepers, :]

### monthly average
one_day = 1  # this code was copied from the observed code which had 48 half-hourly measurements per day
one_year = one_day * 365
total_months = 12 * config.year_range
num_sites = X.shape[0]
num_vars = X.shape[2]
newX = np.empty((num_sites, total_months, num_vars))
for year in range(config.year_range):
    for month in range(12): 
        start = (year*one_year)  # year
        if month > 0:
            start += np.sum(config.days_per_month[:month]) * one_day  # month
        length_current_month = config.days_per_month[month] * one_day
        end = start + length_current_month
        month_index = year*12 + month
        for site in range(num_sites):
            data = X[site,start:end,:]
            newX[site,month_index,:] = np.nanmean(data, axis = 0)
#
X = np.array(newX)


### offset southern hemisphere by 6 months
shift_size = 6  # six months
lat_ind = list(X_vars).index("lat")
for site in range(num_sites):
    lat = np.nanmax(X[site,:,lat_ind])
    if lat < 0:
        shifted = deepcopy(X[site,shift_size:,:])  # new var of shifted data
        X[site,:,:] = np.nan  # replace old data with nans
        X[site,0:-shift_size,:] = deepcopy(shifted)  # shove in shifted data

### separate out Z: site, lat, long
zinds = [list(X_vars).index('long')] + [list(X_vars).index('lat')]
Z = X[:,:,zinds]
Z_vars = np.array(X_vars)[zinds]

X = np.delete(X, zinds, axis=2)
X_vars = np.delete(np.array(X_vars), zinds)

### separate Y and X
yind = list(X_vars).index("ch4")
Y = X[:,:,yind]
Y_vars = X_vars[yind]
X = np.delete(X, yind, axis=2)
X_vars = np.delete(X_vars, yind)

### z-norm
Y = np.expand_dims(Y, axis=-1)

X_stats = np.zeros((X.shape[-1], 2))
Y_stats = np.zeros((1, 2))  

for v in range(len(X_vars)):
    var = np.nanvar(X[:,:,v])
    if var > 0:
        X[:,:,v], X_stats[v,0], X_stats[v,1] = utils.Z_norm(X[:,:,v])

for v in range(1):
    var = np.nanvar(X[:,:,v])
    Y[:,:,v], Y_stats[v,0], Y_stats[v,1] = utils.Z_norm(Y[:,:,v])

### change nans to -9999 for training with GRU
X = np.nan_to_num(X, nan=-9999)
Y = np.nan_to_num(Y, nan=-9999)
m = Y.flatten()
plt.hist(m.astype(float), bins=50)
plt.show()

### write
torch.save({'X': torch.tensor(X),
            'Y': torch.tensor(Y),
            'Z': torch.tensor(Z),
            'X_stats': X_stats,
            'Y_stats': Y_stats,
            'X_vars': X_vars,
            'Y_vars': Y_vars,
            'Z_vars': Z_vars,
            }, path_save_sim)
