# GWFAST
Fisher package for GW cosmology, written in pure Python and using automatic differentiation.

The detail of implementations and results can be found in the papers [1](<>) and [2](<>).

Waveforms are also separatley released as [WF4Py](<https://github.com/CosmoStatGW/WF4Py>).

Developed by [Francesco Iacovelli](<https://github.com/FrancescoIacovelli>) and [Michele Mancarella](<https://github.com/Mik3M4n>).

## Code Organization
The organisation of the repository is the following:

```
GWFAST/GWFAST/
			├── Globals.py 
					Physical constants, positions and duty cycles of existing detectors
			├── utils.py
					Auxiliary functions: angles and time conversions, ...
			├── waveforms.py
					Abstract class WaveFormModel; different sublasses for each wf model - TaylorF2, IMRPhenomD, ...
			├── signal.py
					A class to compute the GW signal in a single detector (L shaped or triangular), the SNR and the Fisher matrix
			├── fisherTools.py
					Covariance matrix and functions to perform sanity checks on the Fisher - condition number, inversion error, marginalization, localization area, plotting tools
			├── network.py
					A class to model a network of detectors with different locations

GWFAST/psds/ 
			Some detector Power Spectral Densities 
			
GWFAST/WFfiles/ 
			Text files needed for waveform computation
						
```

## Summary

* [Usage](https://github.com/CosmoStatGW/WF4Py#Usage)
* [Installation](https://github.com/CosmoStatGW/WF4Py#Installation)
* [Available models](https://github.com/CosmoStatGW/WF4Py#Available-models)
* [Testing](https://github.com/CosmoStatGW/WF4Py#Testing)
* [Bibliography](https://github.com/CosmoStatGW/WF4Py#Bibliography)


## Usage

All details are reported in the accompanying paper [2](<>).

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

## Installation
To install the package without cloning the git repository simply run
```
pip install git+https://github.com/CosmoStatGW/...
```

## Example 
Here we report the cumulative distributions of SNRs and parameter errors for a population of 100'000 BBH events as seen by a network consisting of LIGO, Virgo and KAGRA, ET alone and ET+2CE

![alt text](<https://github.com/CosmoStatGW/.../blob/master/AllCumulBBH.png>)
