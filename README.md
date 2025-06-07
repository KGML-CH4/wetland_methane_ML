# wetland_methane_ML
Project estimating wetland methane emissions using ML and transfer learning.




## Install instructions
```
conda create --name wmml python=3.12.7
conda install mamba -c conda-forge
mamba install -f /requirements.yml
```

## Container
Alternatively, we provide an Apptainer image including the torch and cuda software needed to run the training scripts: zenodo.org/records/15611626.
Example usage:
```
apptainer exec --nv wmml.sif python <training_script>
```

## preprocessing_FLUXNET

Required inputs:                                                                                                               
- half hourly fluxnet data files                                                                                               
- metadata (reformatted)                                                                                                       
- wetland_classification.txt                                                                                                   
- preprocessed_sim.sav

## training scripts
example command:
```
python train.py <working_dir> <test_site_index> <rep>
```

