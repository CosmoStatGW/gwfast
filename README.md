[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.18914.svg)](http://dx.doi.org/10.5281/zenodo.7060236) [![Documentation Status](https://readthedocs.org/projects/gwfast/badge/?version=latest)](https://gwfast.readthedocs.io/en/latest/?badge=latest) [![PyPI version](https://badge.fury.io/py/gwfast.svg)](https://badge.fury.io/py/gwfast) <a href="https://ascl.net/2212.001"><img src="https://img.shields.io/badge/ascl-2212.001-blue.svg?colorB=262255" alt="ascl:2212.001" /></a>[![INSPIRE](https://img.shields.io/badge/INSPIRE-Iacovelli:2022bbs-001529.svg)](https://inspirehep.net/literature/2106524) [![INSPIRE](https://img.shields.io/badge/INSPIRE-Iacovelli:2022mbg-001529.svg)](https://inspirehep.net/literature/2112457)

![alt text](<https://raw.githubusercontent.com/CosmoStatGW/gwfast/master/gwfast_logo_bkgd.png>)

# gwfast
Fisher Information Matrix package for GW cosmology, written in Python and based on automatic differentiation.

The detail of implementations and results can be found in the papers [arXiv:2207.02771](<https://arxiv.org/abs/2207.02771>) and [arXiv:2207.06910](<https://arxiv.org/abs/2207.06910>).

Waveforms are also separatley released as [WF4Py](<https://github.com/CosmoStatGW/WF4Py>).

Developed by [Francesco Iacovelli](<https://github.com/FrancescoIacovelli>) and [Michele Mancarella](<https://github.com/Mik3M4n>).

## Code Organization
The organisation of the repository is the following:

```
gwfast/gwfast/
			├── gwfastGlobals.py 
					Physical constants, positions and duty cycles of existing detectors
			├── gwfastUtils.py
					Auxiliary functions: angles and time conversions, ...
			├── waveforms.py
					Abstract class WaveFormModel; different sublasses for each wf model - TaylorF2, IMRPhenomD, ...
			├── signal.py
					A class to compute the GW signal in a single detector (L shaped or triangular), the SNR and the Fisher matrix
			├── fisherTools.py
					Covariance matrix and functions to perform sanity checks on the Fisher - condition number, inversion error, marginalization, localization area, plotting tools
			├── network.py
					A class to model a network of detectors with different locations
			├── population/
					Modules to perform Fisher forecasts on the accuracy of the reconstruction of the hyperparameters for a population of sources
			├── stochastic/
					Module to copute useful quantities related to stochastic gravitational wave searches

gwfast/psds/ 
			Some detector Power Spectral Densities 
			
gwfast/WFfiles/ 
			Text files needed for waveform computation
			
gwfast/run/
			Scripts to run in parallel on catalogs
			
gwfast/docs/ 
			Code documentation in Sphinx
						
```

## Summary

* [Documentation](https://github.com/CosmoStatGW/gwfast#Documentation)
* [Installation](https://github.com/CosmoStatGW/gwfast#Installation)
* [Usage](https://github.com/CosmoStatGW/gwfast#Usage)
* [Citation](https://github.com/CosmoStatGW/gwfast#Citation)

## Documentation

gwfast has its documentation hosted on Read the Docs [here](<https://gwfast.readthedocs.io/en/latest/>), and it can also be built from the ```docs``` directory.

## Installation
To install the package without cloning the git repository, and a CPU-only version of JAX 

```
pip install --upgrade pip
pip install gwfast
```

or 

```
pip install --upgrade pip
pip install --upgrade "jax[cpu]" 
pip install git+https://github.com/CosmoStatGW/gwfast
```

To install a JAX version for GPU or TPU proceed as explained in [https://github.com/google/jax#installation](<https://github.com/google/jax#installation>).

If willing to use numerical differentiation, a patch has to be applied to [```numdifftools```](<https://pypi.org/project/numdifftools/>). This can be done by running the following command while being in the environment ```gwfast``` has been installed into

```
patch $(python -c "import site; print(site.getsitepackages()[0])")"/numdifftools/limits.py" $(python -c "import site; print(site.getsitepackages()[0])")"/gwfast/.patch/patch_ndt_complex_0-9-41.patch
```


## Usage

All details are reported in the accompanying paper [arXiv:2207.06910](<https://arxiv.org/abs/2207.06910>) and some examples are in the [gwfast_tutorial](<https://github.com/CosmoStatGW/gwfast/blob/master/notebooks/gwfast_tutorial.ipynb>) notebook. <a target="_blank" href="https://colab.research.google.com/github/CosmoStatGW/gwfast/blob/master/notebooks/gwfast_tutorial.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

To initialise a *waveform* object simply run, e.g.

```python
mywf = waveforms.IMRPhenomD()
```
(more details on the waveforms are available in their dedicated git repository [WF4Py](<https://github.com/CosmoStatGW/WF4Py>))

and to build a *signal* object 

```python
MyDet = signal.GWSignal(mywf, psd_path= 'path/to/Detector/psd',
 						detector_shape = 'L', det_lat=43.6, 
 						det_long=10.5, det_xax=115.) 
```

More signal objects can be used to form a *network*

```python
myNet = network.DetNet({'Det1':MyDet1, 'Det2':MyDet2, ...}) 
```

Then computing **SNRs** and **Fisher matrices** is as easy as

```python
SNRs = myNet.SNR(events) 
FisherMatrs = myNet.FisherMatr(events)  
```
where ```events ``` is a dictionary containing the parameters of the chosen events.

Finally, to compute the **covariance matrices** it is sufficient to

```python
CovMatr(FisherMatrs, events) 
```

#### For a list of features implemented after the publication of [arXiv:2207.06910](<https://arxiv.org/abs/2207.06910>) see the [NEW_FEATURES](<https://github.com/CosmoStatGW/gwfast/blob/master/NEW_FEATURES.md>) file and the [new\_features_tutorial](<https://github.com/CosmoStatGW/gwfast/blob/master/notebooks/new_features_tutorial.ipynb>) notebook <a target="_blank" href="https://colab.research.google.com/github/CosmoStatGW/gwfast/blob/master/notebooks/new_features_tutorial.ipynb"> <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Citation

If using this software, please cite this repository and the papers [arXiv:2207.02771](<https://arxiv.org/abs/2207.02771>) and [arXiv:2207.06910](<https://arxiv.org/abs/2207.06910>). Bibtex:

```
@article{Iacovelli:2022bbs,
    author = "Iacovelli, Francesco and Mancarella, Michele and Foffa, Stefano and Maggiore, Michele",
    title = "{Forecasting the Detection Capabilities of Third-generation Gravitational-wave Detectors Using GWFAST}",
    eprint = "2207.02771",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.3847/1538-4357/ac9cd4",
    journal = "Astrophys. J.",
    volume = "941",
    number = "2",
    pages = "208",
    year = "2022"
}
```

```
@article{Iacovelli:2022mbg,
    author = "Iacovelli, Francesco and Mancarella, Michele and Foffa, Stefano and Maggiore, Michele",
    title = "{GWFAST: A Fisher Information Matrix Python Code for Third-generation Gravitational-wave Detectors}",
    eprint = "2207.06910",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.3847/1538-4365/ac9129",
    journal = "Astrophys. J. Supp.",
    volume = "263",
    number = "1",
    pages = "2",
    year = "2022"
}
```