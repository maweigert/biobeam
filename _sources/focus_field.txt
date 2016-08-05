Focus field calculations
========================

The free space focus field (PSF) arising from a given back pupil aperture function can be calculated via the Debye diffraction integral.
*biobeam* can evaluate the diffraction integral for a variety of aperture functions directly on the GPU and thus can calculate the 3D PSF (both scalar and vectorial) extremely fast.   



Gaussian/Bessel beams
---------------------

**Gaussian/Bessel beams**

.. image:: _static/pupil_gauss.png
    :width: 100px
    :align: right

.. image:: _static/pupil_bessel.png
    :width: 100px
    :align: right

.. autofunction:: biobeam.focus_field_beam


Cylindrical Lens
----------------

.. image:: _static/pupil_cylinder.png
    :width: 100px
    :align: right
	   
.. autofunction:: biobeam.focus_field_cylindrical


Bessel Lattices
----------------

.. image:: _static/pupil_lattice.png
    :width: 100px
    :align: right

.. autofunction:: biobeam.focus_field_lattice
   
