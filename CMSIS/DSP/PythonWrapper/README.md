# README

This Python wrapper for CMSIS-DSP is compatible with numpy.

It is a very experimental wrapper with lots of limitations as described in the corresponding section below.

But even with those limitations it can be very useful to test a CMSIS-DSP implemention of an algorithm with all the power oif numpy and scipy.

# How to build and install

The build is using a customized arm_math.h in folder cmsisdsp_pkg/src to be able to compile on windows.

As a consequence, if you build on an Arm computer, you won't get the optimizations of the CMSIS library. It is possible to get them by replacing the customized arm_math.h by the official one.

Following command will build in place.

    > python setup.py build_ext --inplace

If you launch Python from same directory you'll be able to play with the test scripts.

If you want to install the cmsisdsp package, you need to build a binary distribution with 

    > python setup.py bdist

Then it is advised to install it into a virtualenv

For instance, with Python 3:

Create a folder for this virtual environment. For instance : cmsisdsp_tests

Go to this folder.

Type:

    > python -m venv env

Activate the environment:

    > env\Scripts\activate

Install some packages

    > pip install numpy
    > pip instal scipy
    > pip instal matplotplib

Now, you can install the cmsisdsp package:

   > pip install -e "Path To The Folder Containing setup.py"

# Usage

You can looks at testdsp.py and example.py for some examples.

The idea is to follow as close as possible the CMSIS-DSP API to ease the migration to the final implementation on a board.

First you need to import the module

    > import cmsisdsp as dsp

If you use numpy:

    > import numpy as np

Then you can use a CMSIS-DSP function:

    > r = dsp.arm_add_f32(np.array([1.,2,3]),np.array([4.,5,7]))

The function call also be called more simply with

    > r = dsp.arm_add_f32([1.,2,3],[4.,5,7])

But the result of a CMSIS-DSP function will always be a numpy array whatever the arguments were (numpy array or list).

When the CMSIS-DSP function is requiring an instance data structure, it is just a bit more complex:

First we create this instance. 

    > firf32 = dsp.arm_fir_instance_f32()


Although the initialization function on Python side can also be used to initialize some of the fields of the corresponding instance using named arguments it is not advised to do so. In CMSIS-DSP there are init function for this and they can do some additional processing.

So, we need to call the init function

    > dsp.arm_fir_init_f32(firf32,3,[1.,2,3],[0,0,0,0,0,0,0])

The third arument in this function is the state. Since all arguments (except the instance one) are read-only in this Python API, this state will never be changed ! It is just used to communicate the length of the state array which must be allocated by the init function. This argument is required because it is present in the CMSIS-DSP API and in the final C implementation you'll need to allocate a state array with the right dimension.

Since the goal is to be as close as possible to the C API, the API is forcing the use of this argument.

The only change compared to the C API is that the size variables (like blockSize for filter) are computed automatically from the other arguments. This choice was made to make it a bit easier the use of numpy array with the API.

print(firf32.numTaps())
filtered_x = signal.lfilter([3,2,1.], 1.0, [1,2,3,4,5,1,2,3,4,5])
print(filtered_x)

Then you can filter a signal. 

    > print(dsp.arm_fir_f32(firf32,[1,2,3,4,5]))

    The size of this signal should be blockSize. blockSize was inferred from the size of the state array : numTaps + blockSize - 1 according to CMSIS-DSP. So here the signal must have 5 samples.

So if you want to filter more them you can just call the function again. The state variable inside firf32 will ensure that it works like in the CMSIS-DSP C code. 

    > print(dsp.arm_fir_f32(firf32,[1,2,3,4,5]))

If you want vot compare with scipy it is easy bu warning : coefficients for the filter are in opposite order:

    > filtered_x = signal.lfilter([3,2,1.], 1.0, [1,2,3,4,5,1,2,3,4,5])
    > print(filtered_x)

The principles are the same for all other APIs.

For Fourier transforms there are no init functions for the instance variables which must be initialized from a C struct. To make it simpler to use them from Python, the wrapper is introducing its own init functions.

So, for instance:

The signal we want to use for the FFT.
    > nb = 16
    > signal = np.cos(2 * np.pi * np.arange(nb) / nb)

The CMSIS-DSP cfft is requring complex signal with a specific layout in memory.
To remain as close as possible to the C API, we are not using complex numbers in the wrapper. So the complex signal must be converted into a real one. The function imToReal1D is defined in testdsp.py 

    > signalR = imToReal1D(signal)

The we create our FFT instance:

    > cfftf32=dsp.arm_cfft_instance_f32()

We initialize the instance with the init function provided by the wrapper:

    > status=dsp.arm_cfft_init_f32(cfftf32,nb)
    > print(status)

We compute the FFT:

    > resultR = dsp.arm_cfft_f32(cfftf32,signalR,0,1)

We converte back to a complex format to compare with scipy:

    > resultI = realToIm1D(resultR)
    > print(resultI)

For matrix the instances are masked by the Python API. We decided that for matrix only there was no use for making the CMSIS-DSP instance visibles since they contain the same information as the numpy array (samples, width and dimension).

So to use a CMSIS-DSP matrix function, it is very simple:

    > a=np.array([[1.,2,3,4],[5,6,7,8],[9,10,11,12]])
    > b=np.array([[1.,2,3],[5.1,6,7],[9.1,10,11],[5,8,4]])

Numpy result as reference:
    > print(np.dot(a , b))

CMSIS-DSP result:
    > v=dsp.arm_mat_mult_f32(a,b)
    > print(v)

In a real C code, a pointer to a data structure for the result v would have to be passed as argument of the function.

## example.py

This example depends on a data file which can be downloaded here:

https://www.physionet.org/pn3/ecgiddb/Person_87/rec_2.dat


# LIMITATIONS

Due to the high number of functions in the CMSIS-DSP, the first version of the wrapper was generated automatically from a custom script.

Only a subset of the functions has been tested.

It is likely that some problems are present. The API is quite regular in CMSIS-DSP but there are a few exceptions and the generation script is probably not managing all of them.

So, API may crash due to unallocated variable or wrong data conversion.

The generated C code is a first version for bootstrapping the process. Now that this C file exists, the improvement will be done on it rather than on the generation script.
