# README

This is a Python wrapper for the Arm open source [CMSIS-DSP](https://arm-software.github.io/CMSIS_5/DSP/html/index.html) and it is compatible with `NumPy`.

The CMSIS-DSP is available on our [GitHub](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP) or as a CMSIS Pack.

The idea is to follow as closely as possible the C [CMSIS-DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP) API to ease the migration to the final implementation on a board.

The signal processing chain can thus be tested and developed in a Python environment and then easily converted to a C implementation running on a Cortex-M or Cortex-A board.

A tutorial is also available but with less details than this README:
https://developer.arm.com/documentation/102463/latest/

This wrapper is also containing the scripts for the new CMSIS-DSP [Synchronous Data Flow](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP/SDFTools) (SDF) framework.

SDF is also including some nodes to communicate with Modelica using the VHT Modelica blocks developed as part of our [VHT-SystemModeling](https://github.com/ARM-software/VHT-SystemModeling) demos.



# How to build and install

## Tested configurations

The building of this package has been tested on Windows with the Python install from python.org and Microsoft Visual 2017. 

It has also been tested with `cygwin`. In that case, `python-devel` must be installed too. On Mac, it was tested with standard XCode installation.

To run the examples, `scipy` and `matplotlib` must also be installed.

Other configurations should work but the `setup.py` file would have to be improved. 

Python 3 must be used.

## Installing and Building

### Installing

It is advised to do it in a Python virtual environment. Then, in the virtual environment you can just do:

    pip install cmsisdsp

You must have a recent `pip` (to automatically install the dependencies like `NumPy`) and you should have a compiler which can be found by Python when building the package.

DSP examples are available in the [CMSIS-DSP PythonWrapper examples](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP/PythonWrapper/examples) folder.

Synchronous Data Flow examples are available in the [SDF](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP/SDFTools) folder of [CMSIS-DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP) .

You can also install and run it from [Google colab](https://colab.research.google.com/):

This [link](https://colab.research.google.com/github/ARM-software/CMSIS_5/blob/develop/CMSIS/DSP/PythonWrapper/examples/cmsisdsp_tests.ipynb) will open a Jupyter notebook in [Google colab](https://colab.research.google.com/) for testing. This notebook is from the [examples](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP/PythonWrapper/examples) in the CMSIS-DSP GitHub repository.

### Building

It it is not working (because it is not possible for us to test all configurations), you can then try to build and install the wrapper manually.

It is advised to do this it into a virtualenv


Since the [CMSIS-DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP) wrapper is using `NumPy`, you must first install it in the virtual environment. 

    > pip install numpy

Once `NumPy` is installed, you can build the [CMSIS-DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP) python wrapper. Go to folder `CMSIS/DSP`.
(In a previous version you had to go to the `PythonWrapper` folder. Now the `setup.py` is inside the DSP folder).

Now, you can install the cmsisdsp package in editable mode:

    > pip install -e "Path To The Folder Containing setup.py"



## Running the examples

Install some packages to be able to run the examples

    > pip install numpy
    > pip install scipy
    > pip install matplotlib

Depending on the example, you may have to install more packages.

The examples are in the  [CMSIS-DSP PythonWrapper examples](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP/PythonWrapper/examples) folder.

You can test the scripts `testdsp.py` and `example.py` and try to run them from this virtual environment. `example.py` is requiring a data file to be downloaded from the web. See below in this document for the link.

Note that due to the great number of possible configurations (OS, Compiler, Python), we can't give any support if you have problems compiling the `PythonWrapper` on your specific configuration. But, generally people manage to do it and solve all the problems.

# Usage

The idea is to follow as closely as possible the [CMSIS-DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP) API to ease the migration to the final implementation on a board.

First you need to import the module

    > import cmsisdsp as dsp

If you use numpy:

    > import numpy as np

If you use scipy signal processing functions:

    > from scipy import signal

## Functions with no instance arguments

You can use a [CMSIS-DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP) function with numpy arrays:

    > r = dsp.arm_add_f32(np.array([1.,2,3]),np.array([4.,5,7]))

The function can also be called more simply with

    > r = dsp.arm_add_f32([1.,2,3],[4.,5,7])

The result of a [CMSIS-DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP) function will always be a numpy array whatever the arguments were (numpy array or list).

## Functions with instance arguments 

When the [CMSIS-DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP) function is requiring an instance data structure, it is just a bit more complex to use it:

First you need to create this instance:

    > firf32 = dsp.arm_fir_instance_f32()

Then, you need to call an init function:

    > dsp.arm_fir_init_f32(firf32,3,[1.,2,3],[0,0,0,0,0,0,0])

The third argument in this function is the state. Since all arguments (except the instance ones) are read-only in this Python API, this state will never be changed ! It is just used to communicate the length of the state array which must be allocated by the init function. This argument is required because it is present in the [CMSIS-DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP) API and in the final C implementation you'll need to allocate a state array with the right dimension.

Since the goal is to be as close as possible to the C API, the API is forcing the use of this argument.

The only change compared to the C API is that the size variables (like blockSize for filter) are computed automatically from the other arguments. This choice was made to make it a bit easier the use of numpy array with the API.

Now, you can check that the instance was initialized correctly.

    > print(firf32.numTaps())

Then, you can filter with CMSIS-DSP:

    > print(dsp.arm_fir_f32(firf32,[1,2,3,4,5]))

The size of this signal should be `blockSize`. `blockSize` was inferred from the size of the state array : `numTaps + blockSize - 1` according to [CMSIS-DSP.](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP) So here the signal must have 5 samples.

If you want to filter more than 5 samples, then you can just call the function again. The state variable inside firf32 will ensure that it works like in the [CMSIS-DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP) C code. 

    > print(dsp.arm_fir_f32(firf32,[6,7,8,9,10]))

If you want to compare with scipy it is easy but warning : coefficients for the filter are in opposite order in scipy :

    > filtered_x = signal.lfilter([3,2,1.], 1.0, [1,2,3,4,5,6,7,8,9,10])
    > print(filtered_x)

The principles are the same for all other APIs.

## FFT 

Here is an example for using FFT from the Python interface:

Let's define a signal you will use for the FFT.

    > nb = 16
    > signal = np.cos(2 * np.pi * np.arange(nb) / nb)

The [CMSIS-DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP) cfft is requiring complex signals with a specific layout in memory.

To remain as close as possible to the C API, we are not using complex numbers in the wrapper. So a complex signal must be converted into a real one. The function imToReal1D is defined in testdsp.py 

    > signalR = imToReal1D(signal)

Then, you create the FFT instance with:

    > cfftf32=dsp.arm_cfft_instance_f32()

You initialize the instance with the init function provided by the wrapper:

    > status=dsp.arm_cfft_init_f32(cfftf32, nb)
    > print(status)

You compute the FFT of the signal with:

    > resultR = dsp.arm_cfft_f32(cfftf32,signalR,0,1)

You convert back to a complex format to compare with scipy:

    > resultI = realToIm1D(resultR)
    > print(resultI)

## Matrix 

For matrix, the instance variables are masked by the Python API. We decided that for matrix only there was no use for having the [CMSIS-DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP) instance visibles since they contain the same information as the numpy array (samples and dimension).

So to use a [CMSIS-DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP) matrix function, it is very simple:

    > a=np.array([[1.,2,3,4],[5,6,7,8],[9,10,11,12]])
    > b=np.array([[1.,2,3],[5.1,6,7],[9.1,10,11],[5,8,4]])

`NumPy` result as reference:

    > print(np.dot(a , b))

[CMSIS-DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP) result:

    > v=dsp.arm_mat_mult_f32(a,b)
    > print(v)

In a real C code, a pointer to a data structure for the result `v` would have to be passed as argument of the function.

## example.py

This example depends on a data file which can be downloaded here:

https://archive.physionet.org/pn3/ecgiddb/Person_87/rec_2.dat

This signal was created for a master thesis:

Lugovaya T.S. Biometric human identification based on electrocardiogram. [Master's thesis] Faculty of Computing Technologies and Informatics, Electrotechnical University "LETI", Saint-Petersburg, Russian Federation; June 2005. 

and it is part of the PhysioNet database

Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation 101(23):e215-e220 [Circulation Electronic Pages; http://circ.ahajournals.org/cgi/content/full/101/23/e215]; 2000 (June 13). 

## Submodules

The Python wrapper is containing three submodules : `fixedpoint` , `mfcc` and `datatype`

`fixedpoint` is proving some tools to help generating the fixedpoint values expected
by [CMSIS-DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP).

`mfcc` is generating some tools to generate the MEL filters, DCT and window coefficients
expected by the [CMSIS-DSP](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/DSP) MFCC implementation.

MEL filters are represented as 3 arrays to encode a sparse array.

`datatype` is an API on top of `fixedpoint` to provide more reuse when converting between data formats.

The wrapper is now containing the Synchronous Data Flow Python scripts and you should refer the the documentation in `DSP/SDFTools` folder to know how to use those tools.

