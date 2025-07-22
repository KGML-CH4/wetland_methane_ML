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

- download and unzip each/all sites inside the `<working_dir>/Data/` folder


##### Prepping metadata
The following steps assume the `<working_dir>` is "temp_ch4/" but you can use a different folder:
```
mkdir -p temp_ch4/Data/
ln -s ~/Software/wetland_methane_ML/Data/FLX_AA-Flx_CH4-META_20201112135337801132_reformatted.csv $PWD/temp_ch4/Data/
ln -s ~/Software/wetland_methane_ML/Data/wetland_classification.txt $PWD/temp_ch4/Data/
```

##### Running preprocessing.py
```
python wetland_methane_ML/preprocess_fluxnet.py <working_dir>
```

The `<working_dir>` could be "temp_ch4/", for example.
I usually ask for 50Gb RAM for this step.










## Training
Usage:
```
python <training_script>.py <working_dir> <test_site_index> <rep>
```

`<training_script>` could be `train_baselineML.py` for example.
For each model, this gets repeated for 43 `<test_site_index>`'s, and I do 100 reps for each held out site---both are 0-indexed.




## Model evaluation
Usage:
```
python evaluate.py <working_dir> <model_output_dir> "<plot_title>"
```

The `<model_output_dir>` argument points to the specific folder that gets created for each training script; for example, for `train_baselineML.py` <model_output_dir> should be "<working_dir>/Out/Baseline_ML/".
Last, `<plot_title>` should be in quotes if it contains spaces between words.