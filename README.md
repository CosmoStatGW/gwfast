# gwfast
Fisher Information Matrix package for GW cosmology, written in Python and based on automatic differentiation.

The detail of implementations and results can be found in the papers [1](<>) and [2](<>).

Waveforms are also separatley released as [WF4Py](<https://github.com/CosmoStatGW/WF4Py>).

Developed by [Francesco Iacovelli](<https://github.com/FrancescoIacovelli>) and [Michele Mancarella](<https://github.com/Mik3M4n>).

## Code Organization
The organisation of the repository is the following:

```
gwfast/gwfast/
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

gwfast/psds/ 
			Some detector Power Spectral Densities 
			
gwfast/WFfiles/ 
			Text files needed for waveform computation
			
gwfast/runs/
			Script to run in parallel on catalogs
						
```

## Summary

* [Usage](https://github.com/CosmoStatGW/gwfast#Usage)
* [Installation](https://github.com/CosmoStatGW/gwfast#Installation)
* [Citation](https://github.com/CosmoStatGW/gwfast#Citation)


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
To install the package without cloning the git repository 
```
pip install git+https://github.com/CosmoStatGW/gwfast
```

## Citation

If using this software, please cite this repository and the papers [1](<>) and [2](<>).
