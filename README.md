# wetland_methane_ML
Estimating wetland methane emissions using transfer learning.




## Install instructions
```
conda env create -f requirements.yml
```

## Container
Alternatively, we provide a fully contained and compiled Apptainer image including the torch and cuda software for training ML models: zenodo.org/records/15611626.
Usage:
```
wget https://zenodo.org/records/15611626/files/wmml.sif
apptainer exec --nv --bind </host/path/>:</container/path/> wmml.sif python <training_script>
```








## preprocessing_FLUXNET

Inputs required in the specified working/data directoy.

- metadata (reformatted)                                                                                                       

- wetland_classification.txt

- half hourly fluxnet data files


##### Prepping metadata
```
mkdir Data/
ln -s ~/Software/wetland_methane_ML/Data/FLX_AA-Flx_CH4-META_20201112135337801132_reformatted.csv $PWD/Data/
ln -s ~/Software/wetland_methane_ML/Data/wetland_classification.txt $PWD/Data/
```

##### Instructions for downloading FLUXNET data.
- navigate to https://fluxnet.org/data/download-data/

- click FLUXNET-CH4 Community Product CC-BY-4.0 Data

- select sites you want (see our paper for the sites we used)

- download and unzip each/all sites











## training scripts
Usage:
```
python train.py <working_dir> <test_site_index> <rep>
```

