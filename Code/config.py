import numpy as np

# general settings
start_year, end_year = 2006,2018
year_range = end_year-start_year+1
days_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]

# GRU settings
timesteps_per_year = 12  # *DO NOT TOUCH*  
timesteps = timesteps_per_year*year_range
num_windows = int(np.floor((timesteps-24)/12+1))  # *DO NOT TOUCH*
nonmissing_required = 4  # *DO NOT TOUCH*
lr_adam = 0.001
bsz_obs = 10
patience=10
factor=0.5
maxepoch=100

