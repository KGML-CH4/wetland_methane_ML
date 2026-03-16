# wetland_methane_ML
Estimating wetland methane emissions using transfer learning.




## Install

```
conda env create -f requirements.yml
conda activate wmml
```

#### Container
Alternatively, we provide a fully contained and compiled Apptainer image including the torch and cuda software for training ML models: zenodo.org/records/15611626.

Usage:
```
wget https://zenodo.org/records/15611626/files/wmml.sif
apptainer exec --nv --bind </host/path/>:</container/path/> wmml.sif python <training_script>
```


## Inputs

##### Instructions for downloading FLUXNET data.
- navigate to https://fluxnet.org/data/download-data/

- click FLUXNET-CH4 Community Product CC-BY-4.0 Data

- select sites you want (see our paper for the sites we used)

- download and unzip each/all sites inside the `<working_dir>/Data/` folder


##### Prepping metadata
The following steps assume the `<working_dir>` is "temp_ch4/" but you can use a different folder:
```
mkdir -p temp_ch4/Data/
ln -s ~/Software/wetland_methane_ML/Data/FLX_AA-Flx_CH4-META_20201112135337801132_reformatted.csv $PWD/temp_ch4/Data/
ln -s ~/Software/wetland_methane_ML/Data/wetland_classification.txt $PWD/temp_ch4/Data/
```




## Workflows

Several models are compared. For each model, a separate Nextflow script is used to pipeline the entire workflow, from data preprocessing, to training, to testing. See `Workflows/`.



