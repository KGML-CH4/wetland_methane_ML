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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
from matplotlib.colors import SymLogNorm
from skimage.measure import block_reduce
import config
import utils

### params
start_year, end_year = config.start_year, config.end_year
year_range = config.year_range

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
                long = float(newline[long_ind])  # spatial coordinate of the grid cell
                lat = float(newline[lat_ind])
                long_cell, lat_cell = utils.coords2index(long, lat)  # convert the lat long to array indices starting from 0 (instead of negatives)
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
                    data *= -1  # flip sign (TEM emissions are negative)
                    
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
    # print(ch4_only.shape, frin.shape)  # (720, 360, 372) (720, 360, 372)
    
    # save indices where ch4 is missing
    missing = np.where(np.isnan(ch4_only))
    
    # divide (note: FRIN is often nan, outputs nan)
    frin /= 10000  # "The sg_frin is the 10000*(fraction of wetland area), so you may want to convert them to actual fraction first before dividing. The unit should be mg m^-2 day^-1 in this way. You can also exclude any numbers out of 2000 after conversion"
    ch4_only /= frin  # this introduces ~400k additional nans where frin=nan

    # change nan to 0
    frin = np.nan_to_num(frin, nan=0)  # FRIN is stupidly -99999.0 where fraction inundated is 0 (changed to nan, above as it's read in)
    ch4_only = np.nan_to_num(ch4_only, nan=0)  # emissions always 0 (although possibly nan) where FRIN == -99999 (I checked)

    # where ch4 was genuinely missing, change back to missing.
    ch4_only[missing] = np.nan
                                                                        
    # replace X with filtered data
    X[:,:,:,0] = ch4_only
    del ch4_only, frin, missing
    gc.collect()
    return X

def load_siteclass(header):
    print("loading site classification")
    data = np.genfromtxt(config.fp_wetland_map, delimiter=',')
    counter = 0
    var = np.full((720, 360, 12*31, 1), np.nan)
    long_ind = header.index("LONG")
    lat_ind = header.index("LAT")
    class_ind = header.index("CLASS")
    area_ind = header.index("AREA")
    
    for row in range(data.shape[0]):
        long = float(data[row, long_ind]) 
        lat = float(data[row, lat_ind]) 
        long_cell, lat_cell = utils.coords2index(long, lat)  # convert the lat long here to array indices starting from 0
        site_class = data[row, class_ind]
        area = float(data[row, area_ind])

        # deal with -99999s — commonly—but not consistently—over water — currently filtering these because don't know what else to do
        if site_class == -99999:
            site_class = np.nan
    
        # deal with (+)99999s — appears to be land a lot of the time (and much more common than any other value)
        elif site_class == 99999:
            site_class = np.nan
    
        # include alluvials because, I will need to apply TEM or a TEM-surrogate for upscaling

        # repeat the data for all time points
        site_class = np.expand_dims(site_class, axis=0)
        site_class = np.repeat(site_class, 12*31, axis=0)
        area = np.expand_dims(area, axis=0)
        area = np.repeat(area, 12*31, axis=0)

        # stuff in
        var[long_cell, lat_cell, :, 0] = area
        
    #
    return var


### get grid cell areas
def timeseries(data):
    print("converting to timeseries")
    var = []
    ch4_ind = X_vars.index("ch4")
    for long in range(720):
        for lat in range(360):
            datum = data[long, lat, :, :]  # (372, 4)
            # if not np.all(np.isnan(datum[:,class_ind])):  # ********* IMPORTANT TO SKIP THIS FOR WAD2M **********
            long_real, lat_real = utils.index2coords(long,lat)  # convert back to real lat long
            longlat = np.array([long_real, lat_real])  # repeat the data for all time points
            longlat = np.expand_dims(longlat, axis=0)
            longlat = np.repeat(longlat, 12*31, axis=0)        
            datum = np.concatenate([datum, longlat], axis=1)  # add long lat to data
            var.append(np.array(datum))
    #
    var = np.array(var)
    return var

def load_LE(fp):
    print("loading LE", fp)
    data = np.full((720, 360, 1, 31), np.nan)
    nc_file = nc.Dataset(fp)      
    LE = nc_file['slhf']  # valid_time(1), latitude(1801), longitude(3600) 
    LE = np.array(LE).squeeze()#.astype(float)  (1801, 3600)
    LE *= -1  # flip sign —need a reference for this still **********
    seconds = 60*60*24  # seconds per DAY ("monthly means of daily means"—https://confluence-test-dc.ecmwf.int/display/CKB/ERA5%3A%2Bdata%2Bdocumentation?)
    LE /= seconds

    # convert to 0.5 degrees
    ratio = int(LE.shape[1] / 720)

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

    return data

### load NEW wetland types
nc_file = nc.Dataset(config.fp_wetland_type)  
wc = nc_file['wetlandtype']  # (360, 720)
wc = np.array(wc)

# fill in for every day
wetland_class = np.full((720, 360, 12*31, 1), np.nan)
wc = wc.T  # (720, 360)
wc = np.expand_dims(wc, axis=(2, 3))  # (720, 360, 1, 1)
wetland_class = np.broadcast_to(wc, (720, 360, 12*31, 1))
print(wetland_class.shape)  

### load CH4 INTENSITY
def load_intensity(fp, target_year):
    print("loading ch4 intensity", fp)
    data = np.full((720, 360, 12*31), np.nan)
    nc_file = nc.Dataset(fp)  
    intensity = nc_file['CH4_emission']
    intensity = np.array(intensity)  # (360, 720, 12, 31)
    
    # reshape to days per year
    intensity = np.swapaxes(intensity, 0, 1)  # (720, 360, 12, 31)
    intensity = np.reshape(intensity, (720, 360, 12*31))  # (720, 360, 372)
        
    return intensity

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

### load 0.25 degree FRIN
def load_newFRIN(fp, target_year):
    print("loading new FRIN", fp)
    data = np.full((720, 360, 12*31), np.nan)
    nc_file = nc.Dataset(fp)  
    
    frin = nc_file['Fw']   
    frin = np.array(frin)  # (252, 720, 1440)
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
    start_month = 12 * (target_year - 2000)   # despite what the file attributes says, I'm assuming this is months since 2000
    end_month = start_month + 12
    frin = frin[start_month:end_month,:,:]    
    
    # fill in every day in month
    for month in range(12):
        datum = frin[month, :, :]
        datum = np.swapaxes(datum, 0, 1)  # (720, 360)
        datum = np.expand_dims(datum, axis=-1)  # (720, 360, 1)
        datum = np.repeat(datum, 31, axis=-1)  # (720, 360, 31)
        start_day = month*31
        end_day = start_day + 31
        data[:,:,start_day:end_day] = datum

    data = np.expand_dims(data, axis=-1)            
    return data


### read data
X_vars = ["ch4", "LE", "temp", "area", "class", "long", "lat"] 
X = []
for year in range(start_year, end_year+1):
    print(year, flush=True)
    variables = []

    # ch4 INTENSITY
    fpath = config.fp_intensity + "/CH4_emission_intensity_" + str(year) + ".nc"
    var = load_intensity(fpath, year)
    var = np.expand_dims(var, axis=-1)
    
    # scale by NEW frin
    fpath = config.fp_WAD2M_map
    NEWFRIN = load_newFRIN(fpath, year)  # (important to rename variable here because of the above if/else statement)
    var *= NEWFRIN
    variables.append(var)  # add the updated-ch4 to list

    # LE (era5)
    var = []
    for month in range(12):
        fpath = config.fp_le + '/era5_slhf_monthly_' + str(year) + "_" + str(month+1).zfill(2) + '.nc'  # original
        var.append(load_LE(fpath))
    var = np.concatenate(var, axis = 2)
    var = np.reshape(var, (720, 360, 372))
    var = np.expand_dims(var, axis=-1)
    variables.append(var)

    # temperature
    fpath = config.fp_tair
    header = ["LONG", "LAT", "VAR_NAME", "DONTKNOW", "YEAR", "MONTH", "SUM", "min", "mean", "max"]
    T = load_predictor(fpath, header, year)
    variables.append(T)
        
    # grid cell area
    header = ["LONG", "LAT", "WET", "AREA", "CLASS", "CONTINENT"]
    var = load_siteclass(header)
    variables.append(var)

    # new wetland class
    variables.append(wetland_class)
    
    # reshape into time series for each site/location — simultaneously filter for wetlands, add on lat, long
    variables = np.concatenate(variables, axis=-1)
    variables = timeseries(variables)
    X.append(variables)
#
del var, NEWFRIN, T
gc.collect()
X = np.concatenate(X, axis=1)

### filter sites with all misssing outputs
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
    #new_vars.append(v+"_"+str(cat))  # this way re-codes the vtype to a new index
    new_vars.append("site_class_"+str(int(cats[cat])))  # original value (intended to be at least)

X = X[:, :, keep_inds]  
X = np.concatenate([X, new_static], axis=2)

X_vars = np.array(X_vars)[keep_inds]  
X_vars = list(X_vars) + new_vars

### filter time series for "real" days (instead of assuming 31 days per month)
keepers = []
for month in range(12):
    for day in range(31):
        if day < days_per_month[month]:
            keepers.append(True)
            # print(month+1, day+1, "T")
        else:
            keepers.append(False)
            # print(month+1, day+1, "F")

keepers *= year_range  # repeat for all years
X = X[:, keepers, :]

### monthly average
one_day = 1  # this code was copied from the observed code which had 48 half-hourly measurements per day
one_year = one_day * 365
total_months = 12 * year_range
num_sites = X.shape[0]
num_vars = X.shape[2]
newX = np.empty((num_sites, total_months, num_vars))
for year in range(year_range):
    for month in range(12): 
        start = (year*one_year)  # year
        if month > 0:
            start += np.sum(days_per_month[:month]) * one_day  # month
        length_current_month = days_per_month[month] * one_day
        end = start + length_current_month
        month_index = year*12 + month
        for site in range(num_sites):
            data = X[site,start:end,:]
            newX[site,month_index,:] = np.nanmean(data, axis = 0)

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
        X[:,:,v], X_stats[v,0], X_stats[v,1] = config.Z_norm(X[:,:,v])

for v in range(1):
    var = np.nanvar(Y[:,:,v])
    Y[:,:,v], Y_stats[v,0], Y_stats[v,1] = config.Z_norm(Y[:,:,v])

### change nans to -9999 for training with GRU
X = np.nan_to_num(X, nan=-9999)
Y = np.nan_to_num(Y, nan=-9999)

###save the data
torch.save({'X': torch.tensor(X),
            'Y': torch.tensor(Y),
            'Z': torch.tensor(Z),
            'X_stats': X_stats,
            'Y_stats': Y_stats,
            'X_vars': X_vars,
            'Y_vars': Y_vars,
            'Z_vars': Z_vars,
            }, config.fp_upscale_prep)
