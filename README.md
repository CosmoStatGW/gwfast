# GWfish
Fisher package for GW cosmology

## Code Organization

```bash
GWfish/GWfish/
			├── Globals.py 
						Physical constants, positions and duty cycles of existing detectors
			├── utils.py
						Auxiliary functions: angles and time conversions, ...
			├── waveforms.py
						Abstract class WaveFormModel; different sublasses for each wf model - newtonian, taylorf2,  ...
			├── signal.py
						A class to compute the GW signal in a single detector (L shaped or triangular), the SNR, the fisher matrix, and covariance
			├── fisherTools.py
					 	Functions to perform sanity checks on the Fisher - condition number, inversion error, marginalization, localization area, plotting tools
			├── network.py
						A class to model a network of detectors with different locations
						
```		


