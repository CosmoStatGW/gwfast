import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='...',
    version='1.0.0',
    author='Francesco Iacovelli, Michele Mancarella',
    author_email='francesco.iacovelli@unige.ch, michele.mancarella@unige.ch',
    description='A fisher matrix python package for GW studies',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/CosmoStatGW/gwfast',
    license='GNU GPLv3',
    python_requires='>=3.7',
    packages=setuptools.find_packages(),
    install_requires=['numpy', 'jax', 'scipy', 'astropy', 'h5py', 'mpmath', 'matplotlib', 'numdifftools', 'schwimmbad'],
)
