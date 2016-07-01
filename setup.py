import os
from setuptools import setup, find_packages

setup(name='biobeam',
    version='0.1',
    description='beam propagation',
    url='https://maweigert.github.io/biobeam',
    author='Martin Weigert',
    author_email='mweigert@mpi-cbg.de',
    license='MIT',
    packages=['biobeam'],

    install_requires=[
        'numpy',
        'scipy',
        "pyopencl>=2016.1",
        'gputools',
    ],

    package_data={"biobeam":['bpm/kernels/*.cl',
                             'focus_field/kernels/*cl'],},

)
