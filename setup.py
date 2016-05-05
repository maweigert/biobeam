import os
from setuptools import setup, find_packages

setup(name='biobeam',
    version='0.1',
    description='beam propagation',
    url='',
    author='Martin Weigert',
    author_email='mweigert@mpi-cbg.de',
    license='MIT',
    packages=['biobeam'],
    install_requires=[
        'numpy', 'scipy',"pyopencl>=2015.2.4"
    ],

    package_data={"biobeam":['bpm/kernels/*.cl']},

)
