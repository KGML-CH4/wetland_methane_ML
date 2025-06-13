# wetland_methane_ML
Estimating wetland methane emissions using transfer learning.




## Install instructions
First install conda, instructions here: https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer

Then,
```
conda env create -f requirements.yml
conda activate wmml
```

## Container
Alternatively, we provide a fully contained and compiled Apptainer image including the torch and cuda software for training ML models: zenodo.org/records/15611626.

Usage:
```
wget https://zenodo.org/records/15611626/files/wmml.sif
apptainer exec --nv --bind </host/path/>:</container/path/> wmml.sif python <training_script>
```






## Preprocessing FLUXNET data

##### Instructions for downloading FLUXNET data.
- navigate to https://fluxnet.org/data/download-data/

- click FLUXNET-CH4 Community Product CC-BY-4.0 Data

- select sites you want (see our paper for the sites we used)

- download and unzip each/all sites


##### Prepping metadata
```
mkdir -p temp_ch4/Data/
ln -s ~/Software/wetland_methane_ML/Data/FLX_AA-Flx_CH4-META_20201112135337801132_reformatted.csv $PWD/temp_ch4/Data/
ln -s ~/Software/wetland_methane_ML/Data/wetland_classification.txt $PWD/temp_ch4/Data/
```

##### Running preprocessing.py
I usually ask for 50Gb RAM for this step.
```
python wetland_methane_ML/preprocess_fluxnet.py temp_ch4
```










## Training
Should work with 20Gb RAM
Usage:
```
python train.py <working_dir> <test_site_index> <rep>
```

