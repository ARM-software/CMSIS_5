.. cmsisdsp documentation master file, created by
   sphinx-quickstart on Mon Feb 14 11:09:26 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CMSIS-DSP PythonWrapper's documentation!
===================================================

**cmsisdsp** is a Python wrapper for the Arm `CMSIS-DSP API <https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP>`_. With **cmsisdsp** you can run the `CMSIS-DSP API <https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP>`_ functions in Python. The functions are compatible with NumPy.

The APIs of the functions in package **cmsisdsp** are as close as possible to the C versions to make it easier to migrate the Python implementation of your code to an equivalent C one running on the embedded target.

In addition to the Python interface to the C functions, there are additional submodules.

* :doc:`fixedpoint.py <fixedpoint>` is providing some tools to help generating the fixedpoint values expected
  by CMSIS-DSP.
* :doc:`mfcc.py <mfcc>` is generating some tools to generate the MEL filters, DCT and window coefficients
  expected by the CMSIS-DSP MFCC implementation.

Table of contents
*****************
.. toctree::

   Python API to C functions <api>
   datatype.py <datatype>
   fixedpoint.py <fixedpoint>
   mfcc.py <mfcc>

.. note::

   This project is under development.