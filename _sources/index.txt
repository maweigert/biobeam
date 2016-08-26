biobeam - Fast simulation of image formation for in-silico tissue microscopy
============================================================================
.. 
    .. image:: _static/droso_rotate.gif

	   
.. image:: _static/prop_composite_cycle.gif
    :width: 600px
    :align: center

*biobeam* is an open software platform to rigorously simulate the image-formation process of fluorescent light microscopes, especially deep inside scattering biological tissues.

It is designed to provide a fast and easy to use API for computational micoscopy and includes modules for

- scalar and vectorial PSF calculations

- fast GPU-acclerated beam propagation of arbitrary light fields through a given refractive index map

- the creation of simulated g3d image datasets from ground truth while taking for the whole image formation process of light-sheet microscopy into account.
All modules use GPU-acceleration via OpenCL, making all these calculations exceedingly fast. 



			  
.. toctree::
   :hidden:
   :numbered:
   :maxdepth: 2

   intro
   installing
   basic
   beams
   focus_field
   forward_model
   aberrations
   examples
			  
   	 
