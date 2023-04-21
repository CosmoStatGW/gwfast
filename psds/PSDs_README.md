# gwfast/psds
We here list the sources of the available Power Spectral Densities, PSDs, or Amplitude Spectral Densities, ASDs available in GWFast, in alphabetical order

### ce\_strain/

Cosmic Explorer ASDs from [*Science-Driven Tunable Design of Cosmic Explorer Detectors*](https://arxiv.org/abs/2201.10668), available at [https://dcc.cosmicexplorer.org/cgi-bin/DocDB/ShowDocument?.submit=Identifier&docid=T2000017&version=](https://dcc.cosmicexplorer.org/cgi-bin/DocDB/ShowDocument?.submit=Identifier&docid=T2000017&version=).

The folder contains ASDs for:

* the baseline 40km detector (```cosmic_explorer```)
* the baseline 20 km detector compact binary tuned (```cosmic_explorer_20km```)
* the 20 km detector tuned for post-merger signals (```cosmic_explorer_20km_pm```)
* the 40 km detector tuned for low-freqency signals (```cosmic_explorer_40km_lf```)

### ET\_designs\_comparison\_paper/

Einstein Telescope ASDs from [*Science with the Einstein Telescope: a comparison of different designs*](https://arxiv.org/abs/2303.15923), available at [https://apps.et-gw.eu/tds/?content=3&r=18213](https://apps.et-gw.eu/tds/?content=3&r=18213).

The folder contains two subfolders with ASDs for:

* the high frequency (HF) only ET instrument with a length of 10 km (```ETLength10km```), 15 km (```ETLength15km```) and 20 km (```ETLength20km```), in the **HF_only/** folder
* the full high frequency (HF) and low frequency (LF) ET instrument in the cryogenic design with a length of 10 km (```ETLength10km```), 15 km (```ETLength15km```) and 20 km (```ETLength20km```), in the **HFLF_cryo/** folder


### ET-0000A-18.txt

Public [ET-D](https://arxiv.org/abs/1012.0908) sensnitivity curve. 

Available at [https://apps.et-gw.eu/tds/?content=3&r=14065](https://apps.et-gw.eu/tds/?content=3&r=14065). Notice that we kept only the first and last column of the file, corresponding to the frequencies and the total ET-D sensitivity, obtained combining the LF and HF instruments.

### LVC_O1O2O3/

The folder contains ASDs for the LIGO and Virgo detectors during their O1, O2 and O3 observing runs, extracted in specific moment from actual data.

Available at [https://dcc.ligo.org/P1800374/public/](https://dcc.ligo.org/P1800374/public/) for O1 and O2, [https://dcc.ligo.org/LIGO-P2000251/public](https://dcc.ligo.org/LIGO-P2000251/public) for O3a, and computed using [PyCBC](https://pycbc.org) around the times indicated in the caption of Fig. 2 of [https://arxiv.org/abs/2111.03606](https://arxiv.org/abs/2111.03606).

### observing\_scenarios\_paper/

ASDs used for the paper [*Prospects for observing and localizing gravitational-wave transients with Advanced LIGO, Advanced Virgo and KAGRA*, KAGRA Collaboration, LIGO Scientific Collaboration and Virgo Collaboration](https://link.springer.com/article/10.1007/s41114-020-00026-9).

Available at [https://dcc.ligo.org/LIGO-T2000012/public](https://dcc.ligo.org/LIGO-T2000012/public). 

The folder contains ASDs for the Advanced LIGO, Advanced Virgo and KAGRA detectors during the O3, O4 and O5 observing runs.

### unofficial\_curves\_all\_dets/

Public ASDs for both the current and future generation of detectors (last update in January 2020). 

Available at [https://dcc.ligo.org/LIGO-T1500293/public](https://dcc.ligo.org/LIGO-T1500293/public), in the *curves\_Jan\_2020.zip* file.
  
The folder contains ASDs for:

* Advanced LIGO and Advanced Virgo during both the O1, O2 and O3 runs, at design sensitivity and in the *Advanced plus* stage;
* KAGRA;
* LIGO Voyager;
* ET-D;
* CE1 and CE2.
