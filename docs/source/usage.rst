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

.. See this `examples notebook <notebooks/note>` to get started with :mod:`pfm` package or `this one <notebooks/notebook>` to get started with :mod:`probe_station` package.

.. You can also check out :doc:`test` and :doc:`test` for API reference.