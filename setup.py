import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='gwfast',
    version='1.1.0',
    author='Francesco Iacovelli, Michele Mancarella',
    author_email='francesco.iacovelli@unige.ch, michele.mancarella@unige.ch',
    description='A fisher matrix python package for GW studies',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/CosmoStatGW/gwfast',
    license='GNU GPLv3',
    python_requires='>=3.7',
    packages=['gwfast', 'run'],#setuptools.find_packages(),
    include_package_data=True,
    package_data={'':['../WFfiles/*.txt', '../WFfiles/*.h5', '../psds/*', '../psds/ce_curves/*', '../psds/LVC_O1O2O3/*', '../psds/observing_scenarios_paper/*', '../psds/unofficial_curves_all_dets/*']},
    install_requires=['numpy', 'scipy', 'astropy', 'h5py', 'mpmath', 'matplotlib', 'numdifftools', 'schwimmbad'],
)
