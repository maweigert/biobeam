Examples
========



Plane wave scattered by sphere
------------------------------

.. code:: python

    from biobeam import Bpm3d
    # create the refractive index difference
    x = 0.1 * np.arange(-128,128)
    Z, Y, X = np.meshgrid(x,x,x,indexing = "ij")
    R = np.sqrt(X**2+Y**2+Z**2)
    dn = 0.05*(R<2.)

    # create the computational geometry
    m = Bpm3d(dn = dn, units = (0.1,0.1,0.1), lam = 0.5)

    # propagate a plane wave and return the intensity
    u = m._propagate(return_comp = "intens")

    # vizualize
    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.imshow(u[...,128], cmap = "hot")
    plt.title("zy slice")
    plt.subplot(1,2,2)
    plt.imshow(u[128,...], cmap = "hot")
    plt.title("xy slice")


.. figure:: _static/example_sphere.png
   :width: 500px
   :align: center

		   


..
   Light sheet through cell phantom
   --------------------------------


   Computing the psf inside a cell phantom
   ---------------------------------------


   Aberration from sphere
   ----------------------




