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


def random_flip_rotate(image):
    if random.random() > 0.5:
        image = torch.flip(image, dims=[2])
    if random.random() > 0.5:
        image = torch.flip(image, dims=[1])
    k = random.randint(0, 3)
    image = torch.rot90(image, k=k, dims=[1, 2])
    return image


def split_data_group(data0,shuffled_ind,train_frac=0.7,val_frac=0.2,test_frac=0.1):
    sample_size = len(data0)
    train_n=int(train_frac*sample_size)
    if test_frac > 0:
        val_n=int(val_frac*sample_size)
        test_n=sample_size - train_n - val_n
    else:
        val_n = sample_size-train_n
        test_n=0
    #
    data_train, data_val, data_test = [],[],[]
    for i in range(0, train_n):
        data_train.append(data0[shuffled_ind[i]])
    for i in range(train_n, train_n+val_n):
        data_val.append(data0[shuffled_ind[i]])
    data_train = torch.cat(data_train, dim=0)
    data_val = torch.cat(data_val, dim=0)
    data_train = data_train.to(torch.float32)
    data_val = data_val.to(torch.float32)
    if test_frac > 0:
        for i in range(val_n, sample_size):
            data_test.append(data0[shuffled_ind[i]])
        #
        data_test = torch.cat(data_test, dim=0)
        data_test = data_test.to(torch.float32)
  
    return data_train,data_val,data_train



