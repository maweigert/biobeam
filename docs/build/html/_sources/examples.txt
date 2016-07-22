Examples
========


Plane wave scattered by sphere
------------------------------

.. code:: python

    # create the refractive index difference
    x = 0.1 * np.arange(-128,128)
    Z, Y, X = np.meshgrid(x,x,x,indexing = "ij")
    R = np.sqrt(X**1+Y**2+Z**2)
    dn = 0.05*(R<2.)

    # create the computational geometry
    m = Bpm3d(dn = dn, units = (0.1,0.1,0.1), lam = 0.5)

    # propagate a plane wave and return the intensity
    u = m._propagate()

    # vizualize
    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.imshow(u[...,128], cmap = "hot")
    plt.title("zy slice")
    plt.subplot(1,2,2)
    plt.imshow(u[128,...], cmap = "hot")
    plt.title("xy slice")



Light sheet through cell phantom
--------------------------------


Computing the psf inside a cell phantom
---------------------------------------



Aberration from sphere
----------------------




.. include:: example_simple.rst
