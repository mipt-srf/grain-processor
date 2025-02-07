Usage
=====

You can start with this snippet of code

.. code-block:: python

    from grain_processor import GrainProcessor as GP
    gp = GP(
        r"path\to\image.tif",
        cut_SEM=True,
        fft_filter=True,
    )
    gp.save_results()
