Contribution
============

Clone the repository

.. code-block:: console

    $ git clone https://github.com/mipt-srf/grain-processor

Open repository folder

.. code-block:: console

    $ cd grain-processor 

Install package in editable mode (via pip or uv)

.. tab-set::

    .. tab-item:: pip

        Create virual environment

        .. code-block:: console

            $ py -m venv .venv

        Activate the environment

        .. code-block:: console

            $ .venv\Scripts\activate

        Install package

        .. code-block:: console

            (.venv) $ pip install -e .

    .. tab-item:: uv

        Create virual environment and install package in editable mode

        .. code-block:: console

            $ uv sync
