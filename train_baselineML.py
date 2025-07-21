import numpy as np
import matplotlib.pyplot as plt
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import sys
from copy import deepcopy
# helper functions
import config
import utils
import helper_train
from nn_architectures import pureML_GRU


### file paths
wd = sys.argv[1]
test_ind = int(sys.argv[2])
rep = int(sys.argv[3])

finetune_path = wd + "/Out/finetune_baselineML_" + str(test_ind) + "_" + str(rep) + ".sav"
#path_out = wd + '/Out/output_baseline_ML_' + str(test_ind) + "_" + str(rep) + ".txt"

sys.stderr.write("using working dir:" + wd + "\n")



### load params                                          
start_year, end_year = config.start_year, config.end_year
days_per_month = config.days_per_month
timesteps_per_year = config.timesteps_per_year
num_windows = config.num_windows
nonmissing_required = config.nonmissing_required
lr_adam=config.lr_adam
bsz_obs = config.bsz_obs
patience=config.patience
factor=config.factor
maxepoch=config.maxepoch



### load observed data
fp = wd + "/Out/preprocessed_obs.sav"
data0 = torch.load(fp, weights_only=False)
X_obs = data0['X']
Y_obs = data0['Y']
Z_obs = data0['Z']
print(X_obs.shape)
print(Y_obs.shape)
print(Z_obs.shape)
X_vars_obs = data0['X_vars']
Y_vars_obs = data0['Y_vars']
print(Y_vars_obs, flush=True)
Z_vars_obs = data0['Z_vars']
X_stats = data0['X_stats']
Y_stats = data0['Y_stats']
print(X_vars_obs, flush=True)
#print(Y_vars, flush=True)
print(Z_vars_obs, flush=True)
print(X_vars_obs, flush=True)
print(X_obs.shape, flush=True)
print(Y_obs.shape, flush=True)
print(Z_obs.shape, flush=True)

# site ID 
ids = np.arange(X_obs.shape[0])  # this notebook is splitting by site ID—no option for random split
print(len(set(list(ids))), "sites")



### Separate out F_CH4 (into variable "M")

# create new var
mind = list(X_vars_obs).index("FCH4")
M_obs = deepcopy(X_obs[:,:,mind])
M_vars_obs = deepcopy(X_vars_obs[mind])
M_stats = deepcopy(X_stats[mind, :])
print(M_vars_obs, M_obs.shape, M_stats)

# remove from X
X_obs = np.delete(X_obs, mind, axis=2)
X_vars_obs = np.delete(X_vars_obs, mind)
X_stats = np.delete(X_stats, mind, axis=0)

print(X_obs.shape)
print(X_vars_obs)
print(X_stats.shape)



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



### prep and filter time windows
print(X_obs.shape)
print(Y_obs.shape)
print(Z_obs.shape)
print(M_obs.shape)
print(G_obs.shape)
 
# chunk up train sites into windows
new_X, new_Y, new_Z, new_M, new_G = [],[],[],[],[]
num_sites = X_obs.shape[0]
for site in range(num_sites):
    new_site_x, new_site_y, new_site_z, new_site_m, new_site_g = [],[],[],[],[]
    for it in range(num_windows):
        window_range_in = range(timesteps_per_year*it,timesteps_per_year*it+(timesteps_per_year*2))  # window of (smaller) input
        x_piece = X_obs[site, window_range_in, :]#.to(device)
        y_piece = Y_obs[site, window_range_in, 0]#.to(device)
        z_piece = Z_obs[site, window_range_in, :]#.to(device)
        m_piece = M_obs[site, window_range_in]#.to(device)
        g_piece = G_obs[site, window_range_in]#.to(device)
        # print(x_piece.shape, y_piece.shape, z_piece.shape)  # torch.Size([24, 6]) torch.Size([24]) torch.Size([24, 3])

        # year 1
        x_piece_1 = x_piece[0:timesteps_per_year, :]
        y_piece_1 = y_piece[0:timesteps_per_year]

        # first check: are inputs consistently missing for each time step? I would it expect the model to get tripped up otherwise.
        # (rather, instead of checking, just assigning missing to all inputs where there is at least one missing)
        x_missing_1 = (x_piece_1 == -9999).any(dim=1)  # check which months have missing inputs for any variable
        for month in range(timesteps_per_year):
            if x_missing_1[month] == True: # if any is missing, replace all with missing
                x_piece_1[month, :] = -9999                        

        # (skipping filter on missing data in year-1, i.e., it can be all missing)
                
        # year 2
        x_piece_2 = x_piece[timesteps_per_year:(timesteps_per_year*2), :]
        y_piece_2 = y_piece[timesteps_per_year:(timesteps_per_year*2)]
        
        # second check: do positions of missing x values == missing y value positions?
        # We cannot expect the model to output a good estimate for, say, January, if there are no inputs.
        # But the reciprocal: we should keep the inputs even if no output, because that's more information for the RNN to work with.
        x_missing_2 = (x_piece_2 == -9999).any(dim=1)
        y_missing_2 = (y_piece_2 == -9999) 
        for month in range(timesteps_per_year):
            if x_missing_2[month] == True:
                x_piece_2[month, :] = -9999  # if any X's are missing, set the rest to missing to avoid confusing the model
                y_piece_2[month] = -9999
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

X_obs_windows = list(new_X)
Y_obs_windows = list(new_Y)
Z_obs_windows = list(new_Z)
M_obs_windows = list(new_M)
G_obs_windows = list(new_G)
print(len(X_obs_windows), len(Y_obs_windows), len(Z_obs_windows), len(M_obs_windows), len(G_obs_windows))
print(X_obs_windows[0].shape, Y_obs_windows[0].shape, Z_obs_windows[0].shape, M_obs_windows[0].shape, G_obs_windows[0].shape)



### train/val/test split

# separate single test site
###test_ind = 0
X_test = X_obs_windows[test_ind]
Y_test = Y_obs_windows[test_ind]
Z_test = Z_obs_windows[test_ind]
M_test = M_obs_windows[test_ind]
print(X_test.size(), Y_test.size(), Z_test.size(), M_test.size(), flush=True)
if len(X_test) == 0:
    print("\n\n\n\t test data empty\n\n\n")
    sys.exit()

# shuffle remaining
X_temp = deepcopy(X_obs_windows)
Y_temp = deepcopy(Y_obs_windows)
Z_temp = deepcopy(Z_obs_windows)
M_temp = deepcopy(M_obs_windows)
del X_temp[test_ind]
del Y_temp[test_ind]
del Z_temp[test_ind]
del M_temp[test_ind]
shuffled_ind = torch.randperm(len(X_temp))

# X
trainf=0.7; vf=0.3; testf=0
X_train, X_val, _ = helper_train.split_data_group(X_temp,shuffled_ind, train_frac=trainf, val_frac=vf, test_frac=testf)
print(X_train.size(), X_val.size(), flush=True)

# Y
Y_train, Y_val, _  = helper_train.split_data_group(Y_temp,shuffled_ind, train_frac=trainf, val_frac=vf, test_frac=testf)
print(Y_train.size(), Y_val.size(), flush=True)

# Z
Z_train, Z_val, _  = helper_train.split_data_group(Z_temp,shuffled_ind, train_frac=trainf, val_frac=vf, test_frac=testf)
print(Z_train.size(), Z_val.size(), flush=True)

# M
M_train, M_val, _  = helper_train.split_data_group(M_temp,shuffled_ind, train_frac=trainf, val_frac=vf, test_frac=testf)
print(M_train.size(), M_val.size(), flush=True)



### initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device, flush=True)

dropout = 0
model = pureML_GRU(len(X_vars_obs))
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
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=factor)

# initialize some varts
loss_val_best = 500000
best_epoch = 9999
train_losses = []
val_losses = []



### train
train_n = X_train.size(0)
val_n = X_val.size(0)
test_n = X_test.size(0)
X_val = X_val.to(device)
Y_val = Y_val.to(device)

for epoch in range(maxepoch):
    train_loss=0.0
    val_loss=0.0
    shuffled_b=torch.randperm(X_train.size(0)) 
    X_train_shuff=X_train[shuffled_b,:,:] 
    Y_train_shuff=Y_train[shuffled_b,:,:]
    
    # forward
    model.train()  
    model.zero_grad()
    for bb in range(int(train_n/bsz_obs)):
        if bb != int(train_n/bsz_obs)-1:
            sbb = bb*bsz_obs
            ebb = (bb+1)*bsz_obs
        else:
            sbb = bb*bsz_obs
            ebb = train_n
        hidden = model.init_hidden(ebb-sbb).to(device)
        optimizer.zero_grad()
        X_input = X_train_shuff[sbb:ebb, :, :].to(device)
        Y_true = Y_train_shuff[sbb:ebb, :, :].to(device)

        #print(np.sum(np.array(np.isnan(X_input))))
        #print(np.sum(np.array(np.isnan(Y_true))))
        #print(X_input)
        Y_est, _ = model(X_input,hidden)

        Y_est = Y_est[:,timesteps_per_year:(timesteps_per_year*2),:]  # chopping off the first year, that was spin-up
        Y_true = Y_true[:,timesteps_per_year:(timesteps_per_year*2),:]
        
        loss = utils.mse_missing(Y_est, Y_true)
        # print(loss)
        # sys.exit()
        hidden.detach_()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_loss += loss.item() * (ebb-sbb)  # (NOT "bsz_obs", since some batches aren't full)
    #
    
    # validation
    model.eval()  
    with torch.no_grad():

        # finalize training loss         
        train_loss /= train_n
        train_losses.append(train_loss)
        
        hidden = model.init_hidden(X_val.shape[0]).to(device)
        Y_val_pred_t, _ = model(X_val,hidden)
        Y_val_pred_t = Y_val_pred_t[:,timesteps_per_year:(timesteps_per_year*2),:]  # chopping off the first year, that was spin-up
        
        loss = utils.mse_missing(Y_val_pred_t, Y_val[:,timesteps_per_year:(timesteps_per_year*2),:])
        val_loss += loss.item() * timesteps_per_year * X_val.shape[0]
        #
        val_loss /= (val_n*timesteps_per_year)        
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        for param_group in optimizer.param_groups:
            print(f"Learning rate after epoch {epoch+1}: {param_group['lr']}")
            
        # save model, update LR
        if val_loss < loss_val_best:
            loss_val_best=np.array(val_loss)
            best_epoch = epoch
            torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'loss': train_loss,
                    'los_val': val_loss,
                    }, finetune_path)   
        print("finished training epoch", epoch+1, flush=True)
        print("learning rate: ", optimizer.param_groups[0]["lr"], "train_loss: ", train_loss, "val_loss:", val_loss, "best_val_loss:",loss_val_best, flush=True)
        path_fs = finetune_path+'fs'
        torch.save({'train_losses': train_losses,
                    'val_losses': val_losses,
                    'model_state_dict_fs': model.state_dict(),
                    }, path_fs)  
#

print("final train_loss:",train_loss,"val_loss:",val_loss,"val loss best:",loss_val_best, flush=True)




### test
test_n = X_test.size(0)
def check_results(total_b, check_xset, check_y1set, Y_stats):
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
        Y1_pred_t, hidden = model(X_input,hidden)

        Y_true = Y_true[:,timesteps_per_year:(timesteps_per_year*2),:]  # chop off first year
        Y1_pred_t = Y1_pred_t[:,timesteps_per_year:(timesteps_per_year*2),:] 
        
        #            
        Y_true_all[sbb:ebb, :, :] = Y_true.to('cpu')  
        Y_pred_all[sbb:ebb, :, :] = Y1_pred_t.to('cpu')  
    #
    loss = utils.mse_missing(Y_pred_all[:,:,0], Y_true_all[:,:,0]).numpy()
    return Y_pred_all, loss


with torch.no_grad():
    checkpoint=torch.load(finetune_path, map_location=torch.device('cpu'), weights_only=False)
    model = pureML_GRU(len(X_vars_obs))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device) #too large for GPU, kif not enough, change to cpu
    model.eval()  # this is "testing" model, it switches off dropout and batch norm.
    epoch = checkpoint['epoch']
    print("epoch", epoch, flush=True)
    Y_test_pred,loss_test =  check_results(test_n, X_test.float(), Y_test, Y_stats)    
    print(loss_test, flush=True)

    # write
    # fp = path_out + str(test_ind) + "_rep_" + str(rep) + '.txt'
    # print(path_out)
    # with open(path_out, "w") as outfile:
    #     for window in range(Y_test_pred.shape[0]):
    #         outline = list(Y_test_pred[window,:,0].numpy())
    #         outline = "\t".join(list(map(str, outline)))
    #         outfile.write(outline + "\n")
    for window in range(Y_test_pred.shape[0]):         
        outline = list(Y_test_pred[window,:,0].numpy())
        outline = "\t".join(list(map(str, outline)))   
        print("FINAL OUT:", outline)
        
