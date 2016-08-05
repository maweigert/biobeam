Simulating light sheet microscopy
=================================


The afore described routines form the biobeam-specific, GPU accelerated basis of wave-optical simulation and accurate field implementations, that enable the assembly of piplines to account for rigorous image-formation process of entire microscopes in tissues.



.. figure:: _static/lightsheet.png
    :width: 500px
    :align: center

    A cylindrical light sheet is scanned through a synthetic embryo.



Next we describe here the steps required to mimic a fluorescent light-sheet microscope. As in practice, its perfomance is determined by the complex interplay of light scattering on the illumination and detection side.

..
   While the reasoning behing the application is described in [Weigert et al 2016 tbc], we focus here on the modular assembly of the pipe-line, which is achievable in very few steps.


