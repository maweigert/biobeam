Installation
============

Requirements
------------

* Python 2.7/3.5+
* a working OpenCL environment
* a sufficiently modern GPU (>2GB memory)

*Mac*

OpenCL should be provided by default :)

*Linux*

e.g. for nvidia cards, install the latest drivers and then the opencl lib/headers

::

   sudo apt-get install opencl-header  nvidia-libopencl1-35 nvidia-opencl-icd-352
   sudo modprobe nvidia-352-uvm


until clinfo shows your GPU as a valid OpenCL device:

::

   sudo apt-get install clinfo
   sudo clinfo

*Windows*

Install your the SDK of your GPU vendor.

Installing *biobeam*
--------------------


A simple 

::

   pip(2|3) install biobeam

should be enough to get you the package.



To nicely render the 3d output just install the OpenCL accelerated renderer `Spimagine <https://github.com/maweigert/spimagine>`_ 

::

   pip(2|3) install spimagine
