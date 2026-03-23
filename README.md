# ml-flim-imd
Implementation of ML prediction of fluorescence lifetime, destriping, and intensity-modulation pipeline.

## Notes

* The pipeline was created on a Linux system with the following specifications:
    + Ubuntu 24.04.1 LTS 
    + Intel(R) Xeon(R) w7-3465X   2.50 GHz
    + 512 GB RAM
    + NVIDIA RTX A4000 (16GB VRAM)
    + python 3.7+
    + CUDA 12
      
* The pipeline is also compatible with the Janelia compute cluster.

* Installation instructions apply only to the first time this pipeline is run. Afterwards, activating each environment is sufficient.

## Installation

#### To login to the Janelia LSF cluster
```
ssh username@login1.int.janelia.org
```
or login2 if login1 is offline for maintenance

####
These instructions pertain to running the pipeline on the Janelia compute cluster. The instructions are similar for local use, simply omit the "bsub" commands where relevant. If installing on the cluster, wait for completion emails at each step before proceeding.

Verify conda install
```
which conda
```

If conda is not found, install it. We recommend [miniforge](https://github.com/conda-forge/miniforge), following the OS requirements where relevant (Unix for the cluster). 

Clone this reponsitory into your nrs directory by running
```
git clone https://github.com/opp1231/ml-flim-imd
```

**Note: these instructions assume you are working in the directory into which this repository is cloned. If not, preface each filepath with the path to the python folder of this repository.**

### ML Tau Prediction

#### Steps
1. Create the deep-learning FLIM environment
  ```
  bsub -n 1 -W 1:00 "conda create -n dlflim python=3.10"
  ```
2. Activate the environment
  ```
  conda activate dlflim
  ```
3. Install relevant packages
  ```
  bsub -n 1 -W 1:00 "python -m pip install tifffile numpy scipy tensorflow[and-cuda]==2.19"
  ```
4. Deactivate environment
  ```
  conda deactivate
  ```

### Destriping + Intensity-modulated Display (IMD) Projection

#### Steps
1. Create the IMD environment
  ```
  bsub -n 1 -W 1:00 "conda create -n flim_imd python=3.10"
  ```
2. Activate the environment
  ```
  conda activate flim_imd
  ```
3. Install relevant packages
  ```
  bsub -n 1 -W 1:00 "python -m pip install -r ./IMD/requirements.txt"
  ```
4. Deactivate environment
  ```
  conda deactivate
  ```

### Usage

#### ML Tau Prediction
  ```
  bsub -J "EXPERIMENT_NAME_TAU" -W 2:00 -n 4 -gpu "num=1" -q gpu_l4 -o /nrs/path/to/folder/output_%J.txt "python ./src/dlflim_predict.py --arguments"
  ```
  The command-line arguments are passed to the  as follows:

Name | Description | Usage | Default
--- | --- | --- | ---
--inpath | Filepath to the parent folder containing the data | "--inpath /path/to/data/" | *N/A*
--inpath_irf | Filepath to the parent folder containing the IRF | "--inpath /path/to/IRF/" | *N/A*
--inpath_model | Filepath to the parent folder containing the model | "--inpath /path/to/data/" | *N/A*
--gate_size | Gate step in ns | "--gate_size 0.6998" | 0.6998
--nz | Number of z-planes the code should expect | "--nz 5" | 1
--nt | Number of gates in one FLIM stack | "--nt 25" | 30
--version | Predict lifetimes using a single (version=1) or double (version=2) decay model | "--version 2" | 1
--start_t | Which timepoint to be considered the start of the timeseries | "--start_t 2" | 1
--end_t | Which timepoint to be considered the end of the timeseries | "--end_t 200" | -1 (process all timepoints after the start)
--channel | Which channel to process (if two channels exist, channel=0 corresponds to 552 and channel=1 corresponds to 488) | "--channel 1" | 0 

If the value of any argument does not deviate from the default, it can be omitted from the command.
  
#### Destriping + IMD Projection
  ```
  bsub -J "EXPERIMENT_NAME_IMD" -W 2:00 -n 4 -o output_%J.txt "python ./src/projection_IMD.py --arguments"
  ```
  The command-line arguments are as follows:

Name | Description | Usage | Default
--- | --- | --- | ---
--inpath | Filepath to the parent folder containing the data | "--inpath /path/to/data/" | *N/A*
--savepath | Filepath to the folder where output should be saved | "--savepath /path/to/save/folder" | *N/A*
--nz | Number of z-planes the code should expect | "--nz 5" | 1
--nt | Number of gates in one FLIM stack | "--nt 25" | 30
--version | Predict lifetimes using a single (version=1) or double (version=2) decay model | "--version 2" | 1
--min_tau | Lifetime which defines the lowerbound of the colormap | "--min_tau 2.0" | 1.0
--max_tau | Lifetime which defines the upperbound of the colormap | "--max_tau 3.0" | 4.0
--gauss_sig | Sigma for Gaussian filtering of intensity and lifetime images | "--gauss_sig" | 0.5
--noise | Number of wavelets to be used for wavelet-based destriping | "--noise 5" | 10
--sigmay | Filter bandwidth in y (larger gives more filtering) | "--sigmay 32" | 64
--sigmax | Filter bandwidth in x (larger gives more filtering) | "--sigmax 32" | 128
--start_t | Which timepoint to be considered the start of the timeseries | "--start_t 2" | 1
--end_t | Which timepoint to be considered the end of the timeseries | "--end_t 200" | -1 (process all timepoints after the start)
--int_sat | Percent of max intensity which defines saturation | "--int_sat 99.85" | 100.0
--channel | Which channel to process (if two channels exist, channel=0 corresponds to 552 and channel=1 corresponds to 488) | "--channel 1" | 0 
--cmap | Colorcet colormap for intensity-modulated display of lifetime | "--cmap bmy" | rainbow4
--project | Run the sum-projection of intensity | "-p" | False
--destripe | Run the wavelet-based destriping of intensity | "-ds" | False
--imd | Run the intensity-modulation by lifetime | "-im" | False
--total_norm | Normalize intensity over the entire timeseries (false) or frame-by-frame (true -- WARNING: very slow) | "-tn" | False


Note: The IMD pipeline can be run at any point without the "--imd" flag to only sum-project (and potentially destripe) the FLIM stacks. Otherwise, the lifetimes are required for this code to run. As a result, in general, the ML Prediction step should be run before the IMD step.
  


