.. Geodata documentation master file, created by
   sphinx-quickstart on Tue Aug 22 14:57:10 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Geodata's documentation!
===================================

.. include:: intro.rst

.. toctree::
   :maxdepth: 1
   :caption: Quickstart
   :hidden:

   quick_start/packagesetup
   quick_start/input_output

.. toctree::
   :maxdepth: 1
   :caption: Legacy workflow
   :hidden:

   legacy/index

.. toctree::
   :caption: Datasets
   :maxdepth: 1
   :glob:
   :hidden:

   datasets/*

.. toctree::
   :maxdepth: 1
   :caption: Modeling
   :hidden:

   modeling/wind/index
   modeling/pvlib/index

.. toctree::
   :maxdepth: 1
   :caption: Mask
   :glob:
   :hidden:

   mask/*

.. .. toctree::
..    :maxdepth: 1
..    :caption: Parameterization
..    :glob:
..    :hidden:

..    parameterizations/*

.. toctree::
   :maxdepth: 1
   :caption: Visualization
   :glob:
   :hidden:

   visualization/*

.. .. toctree::
..    :maxdepth: 1
..    :caption: Application with Geodata
..    :glob:

..    application/*

.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   development/documentation-organization-plan
   development/offline-era5-fixture-datasets

.. toctree::
   :maxdepth: 1
   :caption: API Reference
   :hidden:

   autoapi/geodata/*


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
