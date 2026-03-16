import torch


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
