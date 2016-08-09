Input Beam patterns 
===================

On a slightly higher abstraction level biobeam allows to use pre-implemented as well as one-line user-definable beam geometries to be simulated. In the following we introduce the relevant function along with a detailed description of their parameters.

.. figure:: _static/beams.png
    :width: 600px
    :align: left

Some pre-define beam types.

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


Cylindrical Lens
----------------

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
   
