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

$${\rm overlap}(h_1, h_2) = \frac{(h_1|h_2)}{\sqrt{(h_1|h_1)(h_2|h_2)}} \quad {\rm with}\quad (h_1|h_2) = 4\,{\rm Re}\int_{f_{\rm min}}^{f_{\rm max}} \frac{\tilde{h}_1(f)\,\tilde{h}_2(f)}{S_n(f)}\,{\rm d}f$$

with $h_i$ being the signal as a function of the parameters, the *tilde* indicates the Fourier transform, $(h_1|h_2)$ denoting the standard scalar product, and $S_n(f)$ being the detector power spectral density (PSD).

This can be computed both from a ```GWSignal``` object or a ```DetNet``` object as 

```python
overlap = myNet.WFOverlap(WF1, WF2, Params1, Params2)
```

with ```WF1``` and ```WF2``` being two waveform objects and ```Params1``` and ```Params2``` being tow dictionaries containing the parameters of the events (as the standard inputs of the ```SNR``` or ```FisherMatr``` functions). This will return an ```array``` with the overlaps of the two waveforms on the two sets of parameters, with the overlap being 1 in case of perfect match.
