# gwfast documentation

## Documentation requirements

In order to build the documentation, the following packages have to be installed

* [```sphinx```](<https://www.sphinx-doc.org/en/master>)
* [```sphinx_rtd_theme```](<https://sphinx-rtd-theme.readthedocs.io/en/stable/>)
* [```nbsphinx```](<https://nbsphinx.readthedocs.io/en/0.8.11/>)
* [```myst-parser```](<https://myst-parser.readthedocs.io/en/latest/>)
* [```sphinx-argparse```](<https://sphinx-argparse.readthedocs.io/en/stable/install.html>)
* [```sphinx-copybutton```](<https://sphinx-copybutton.readthedocs.io/en/latest/?badge=latest>)
* [```readthedocs-sphinx-search```](<https://readthedocs-sphinx-search.readthedocs.io/en/latest/>)
* [```docutils```](<https://docutils.sourceforge.io>)

To install them just run in the terminal 

```
pip install --upgrade pip
pip install -r docs/docs_requirements.txt
```

## Build the documentation

The HTML documentation can easily be built from the ```docs``` folder, running in the terminal 

```
cd docs/
make html
```

The produced ```.html``` files will be stored in the directory ```./build/html```.

It is also possible to build a LaTex version, running in the terminal 

```
make latexpdf
```

the output pdf of this command will be ```./build/latex/gwfast.pdf```.
