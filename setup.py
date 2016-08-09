import os
from setuptools import setup, find_packages

setup(name='biobeam',
    version='0.1.1',
    description='beam propagation',
    url='https://maweigert.github.io/biobeam',
    author='Martin Weigert',
    author_email='mweigert@mpi-cbg.de',
    license='MIT',
    packages=find_packages(),

    install_requires=[
        'numpy',
        'scipy',
        "pyopencl>=2016.1",
        'gputools>=0.1.3',
    ],

    package_data={"biobeam":['core/kernels/*.cl'
                             ],},

)


