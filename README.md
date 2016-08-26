#biobeam - Fast simulation of image formation for in-silico tissue microscopy

<img src="https://github.com/maweigert/biobeam/raw/master/artwork/logo_biobeam_red.png" width="200">

*biobeam* is an open source python package for simulating light propagation on weakly scattering tissue. It implements a scalar beam propagation method while making use of GPU acceleration via PyOpenCL. It is designed to provide fast wave optical simulations common in light sheet microscopy while providing an easy to use API from within Python.

Among the features are

* Fast vectorial psf calculations for various illumination modes (gaussian.bessel beams, cylindrical lenses, bessel lattices...) 
* Fast scalar wave optical simulation of incident light interaction with weakly scattering tissue 
* Simulation module for image formation in light sheet microscopy and aberration calculations

###Quickstart

First make sure there is a valid OpenCL platform available on your machine, e.g. check with 

	clinfo
	
then install everything via pip  

	pip install biobeam


The full documentation with examples can be found [here](https://maweigert.github.io/biobeam).
