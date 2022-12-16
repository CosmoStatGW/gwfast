# Additional features
We here list the features added after the publication of [arXiv:2207.06910](<https://arxiv.org/abs/2207.06910>). 

Developed by [Francesco Iacovelli](<https://github.com/FrancescoIacovelli>) and [Michele Mancarella](<https://github.com/Mik3M4n>).

## Extension of TaylorF2 down to the Kerr ISCO of the remnant BH
The ```TaylorF2_RestrictedPN``` waveform can be extended up to the Kerr ISCO of the remnant BH (rather than assuming it to be a Schwarzschild BH), computed as in [arXiv:2108.05861](<https://arxiv.org/pdf/2108.05861>) (see Appendix C in particular), with the fits from [arXiv:1605.01938](<https://arxiv.org/abs/1605.01938>). This can be done thanks to the ```which_ISCO``` flag as follows:

```python
mywf = waveforms.TaylorF2_RestrictedPN(which_ISCO='Kerr')						
```

## Implementation of a function to compute the waveform overlap

We added a function to compute the **overlap** between two waveforms on two sets of event parameters. This is defined as

$${\rm overlap}(h_1, h_2) = \frac{(h_1|h_2)}{\sqrt{(h_1|h_1)(h_2|h_2)}} \quad {\rm with}\quad (h_1|h_2) = 4 {\rm Re}\int_{f_{\rm min}}^{f_{\rm max}} \frac{\tilde{h}_1(f) \tilde{h}_2(f)}{S_n(f)} {\rm d}f$$

with $h_i$ being the signal as a function of the parameters, the *tilde* indicates the Fourier transform, $(h_1|h_2)$ denoting the standard scalar product, and $S_n(f)$ being the detector power spectral density (PSD).

This can be computed both from a ```GWSignal``` object or a ```DetNet``` object as 

```python
overlap = myNet.WFOverlap(WF1, WF2, Params1, Params2)
```

with ```WF1``` and ```WF2``` being two waveform objects and ```Params1``` and ```Params2``` being tow dictionaries containing the parameters of the events (as the standard inputs of the ```SNR``` or ```FisherMatr``` functions). This will return an ```array``` with the overlaps of the two waveforms on the two sets of parameters, with the overlap being 1 in case of perfect match.

## Addition of the spin-induced quadrupole coefficient due to tidal effects to TaylorF2

The ```TaylorF2_RestrictedPN``` waveform can include the spin-induced quadrupole coefficient as a function the dimensionless tidal deformability, computed as in Eq. (15) of [arXiv:1608.02582](<https://arxiv.org/abs/1608.02582>) with coeffs from third row of Table I. This can be done thanks to the ```use_QuadMonTid``` flag as follows:

```python
mywf = waveforms.TaylorF2_RestrictedPN(is_tidal=True, use_QuadMonTid=True)						
```

Notice the ```is_tidal``` flag has to be set to ```True``` for this to work (the spin-induced quadrupole coefficient for a BH is 1).

## Implementation of a TEOBResumSPA waveform model wrapper

We added a wrapper to use the ```TEOBResumSPA``` waveform model of the ```TEOBResumS``` family, available on Bitbucket [here](<https://bitbucket.org/eob_ihes/teobresums/wiki/Home>), see [arXiv:2104.07533](<https://arxiv.org/abs/2104.07533>), [arXiv:2012.00027](<https://arxiv.org/abs/2012.00027>), [arXiv:2001.09082](<https://arxiv.org/abs/2001.09082>), [arXiv:1904.09550](<https://arxiv.org/abs/1904.09550>), [arXiv:1806.01772](<https://arxiv.org/abs/1806.01772>), [arXiv:1506.08457](<https://arxiv.org/abs/1506.08457>), [arXiv:1406.6913](<https://arxiv.org/abs/1406.6913>). This is a frequency-domain waveform model which can be used both for BBH, BNS and NSBH binaries, including contribution of higher-order modes and tidal effects. It can  be used simply as

```python
mywf = waveforms.TEOBResumSPA_WF()						
```
The flag ```is_tidal=True``` can be used if willing to include tidal effects, and the argument ```modes``` can be used to specify which modes to include in the analysis. It has to be a list of 2-elements lists containig the *l* and *m* of the desired modes (in this order), as e.g.

```python
mywf = waveforms.TEOBResumSPA_WF(modes=[[2,1], [2,2]])						
```
In this case the waveform includes the dominant quadrupole *(l=2, m=2)* mode plus the sub-dominant *(l=2, m=1)* mode. 
The default is to include all the modes up to *l=4*, i.e. ```modes=[[2,1], [2,2], [3,1], [3,2], [3,3], [4,1], [4,2], [4,3], [4,4]]```

The implementation matches the one used in the examples available [here](<https://bitbucket.org/eob_ihes/teobresums/src/master/Python/Examples/>). 

When using this waveform model, derivatives are computed using numerical differentiation (finite differences).

To use this model in the ```calculate_forecasts_from_catalog.py``` script, set ```--wf_model='TEOBResumSPA'``` for the BBH version, or ```--wf_model='TEOBResumSPA_tidal'``` to include tidal effects.
