import numpy as np

# change
wd = "temp_ch4/"
model_version = "/modelStackCNN/"

# general settings
start_year, end_year = 2006,2018
year_range = end_year-start_year+1
days_per_month = [31,28,31,30,31,30,31,31,30,31,30,31]

# GRU settings
timesteps_per_year = 12  
timesteps = timesteps_per_year*year_range
num_windows = int(np.floor((timesteps-24)/12+1))
nonmissing_required = 4
lr_adam = 0.001
bsz_obs = 10
bsz_sim = 1000
patience=10
factor=0.5
maxepoch=100

# Google Earth engine
gee_cred = 'alpine-alpha-435921-p2'

### paths

# inputs
fp_fluxnet_meta = wd + "/Data/FLX_AA-Flx_CH4-META_20201112135337801132_reformatted.csv"  # site-level metadata
fp_wetland_class = wd + "/Data/wetland_classification.txt"
fp_wetland_map = wd + "/Data/wet1x1.tem"
fp_wetland_type = wd + "Data/wetlandtype.nc"
fp_ch4 = wd + "/Data/ch4emi.day"
fp_intensity = wd + "Data/TEM_intensity/"
fp_sgfrin = wd + "/Data/sg_frin.tem"
fp_le = wd + '/Data/LE_land/'
fp_tair = wd + "Data/ecmwf_TAIR_1979-2018.tem"
fp_WAD2M_map = wd + "Data/WAD2M_wetlands_2000-2020_025deg_Ver2.0.nc"

# general preprocessing
fp_fluxnet_raw = wd + "/Out/fluxnet_emissions_HH.csv"  # raw FLUXNET measurements
fp_prep_fluxnet = wd + "/Out/prep_obs.sav"  # preprocessed FLUXNET measurements
fp_modis_fluxnet = wd + "/Out/MODIS_fluxnet/"  # MODIS reflectance over FLUXNET sites
fp_prep_TEM = wd + "/Out/prep_TEM.sav"  # preprocessed TEM data
fp_modis_global = wd + "/Out/MODIS_global/"  # 0.5 degree modis data
fp_modis_intermediate =  wd + "/Out/Intermediate_preprocessing/"
fp_modis_tiles = wd + "/Out/Preprocessed_tiles/"

# model-specific outputs
fp_prep_model = wd + "/Out/" + model_version + "/prep_model.sav"
fp_train = wd + '/Out/' + model_version + '/Training/' 
fp_eval = wd + '/Out/' + model_version + '/Eval/'
fp_upscale_prep = wd + '/Out/' + model_version + '/Upscale/prep_upscale_WAD2M.sav'
fp_upscale_out = wd + '/Out/' + model_version + '/Upscale/'



