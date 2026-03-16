import numpy as np
import torch


def mse_missing(pred, true):
    mask = torch.where(true == -9999, 0, 1)  # 0/1   (e.g. a = torch.tensor([5,-999,3]); torch.where(a == -999, 0, 1))
    non_missing = torch.sum(mask, dim=1)  # sum row-wise: the count of non-missing timepoints per site/training example
    mse = (true-pred)**2  # squared error. Random values for missing data (error = pred-999)
    mse = mse * mask  # missing = 0
    mse = torch.sum(mse, dim=1)  # sum row-wise (by training example)
    mse /= non_missing  # MSE, by training example
    loss = mse.mean()  # mean across training examples
    return loss


def Z_norm_reverse(X,Xnorm,units_convert=1.0):
    return (X*Xnorm[1]+Xnorm[0])*units_convert


def Z_norm(X):
    X_mean=np.nanmean(X)
    X_std=np.nanstd(np.array(X))
    return (X-X_mean)/X_std, X_mean, X_std


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
