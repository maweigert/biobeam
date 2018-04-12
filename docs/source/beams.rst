Input Beam patterns 
===================

To simulate the light field of an incident beam of given type through a given optical medium, the following functions provide the initial conditions of the complex field, which then are propagated through the medium: 


Gaussian/Bessel beams
---------------------

**Gaussian/Bessel beams**

.. image:: _static/pupil_gauss.png
    :width: 100px
    :align: right

.. image:: _static/pupil_bessel.png
    :width: 100px
    :align: right

.. autofunction:: biobeam.focus_field_beam_plane


Cylindrical light sheet
-----------------------

.. image:: _static/pupil_cylinder.png
    :width: 100px
    :align: right
	   
.. autofunction:: biobeam.focus_field_cylindrical_plane


Bessel Lattices
----------------

.. image:: _static/pupil_lattice.png
    :width: 100px
    :align: right

.. autofunction:: biobeam.focus_field_lattice_plane
   
