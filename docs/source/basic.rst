Basic Usage
============

biobeam makes use of highly parallelized propagation of light fields through biological media. What biobeam is unique for is the simulation of the image formation-process in tissue microscopy. For this, the user is free not to care about the details of how different simulations are combined to result in the final 3D images.

Nevertheless, the basic, GPU accelerated wave-optical routines of biobeam remain individually accessible as documented here.

GPU accelerated beam propagation
--------------------------------

*biobeam* uses a GPU compatible beam propagation implementation with the precise, spherical propagator. Most fundamentally the associated routines are accessible by the **Bpm3d** class.

.. code-block:: python

   import biobeam

   dn = ...
   m = Bpm3d(dn = dn, units = (0.1,)*3, lam = 0.5)

   # propagate a plane wave and return the intensity
   u = m.propagate()

   # propagate a bessel beam 
   u = m.propagate(u0 = m.u0_beam(NA = (.4,.42)))
	


.. autoclass:: biobeam.Bpm3d
    :members:
    :special-members: __init__


	   
