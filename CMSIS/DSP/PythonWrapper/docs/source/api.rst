API
===

.. highlight:: python

The idea is to follow as closely as possible the CMSIS-DSP API to ease the migration to the final implementation on a board.

First you need to import the module::

    import cmsisdsp as dsp

If you use numpy::

    import numpy as np

If you use scipy signal processing functions::

    from scipy import signal


Standard APIs
*************

.. code-block:: C
    
     void arm_add_f32(
            const float32_t * pSrcA,
            const float32_t * pSrcB,
            float32_t * pDst,
            uint32_t blockSize);

.. py:function:: dsp.arm_add_f32(pSrcA,pSrcB)

   Return a list of random ingredients as strings.

   :param pSrcA: array.
   :type pSrcA: NumPy array
   :param pSrcB: array.
   :type pSrcB: NumPy array
   :return: array.
   :rtype: NumPy array

Example::

      import cmsisdsp as dsp

      r = dsp.arm_add_f32([1.,2,3],[4.,5,7])

You can use a CMSIS-DSP function with numpy arrays: ::

   r = dsp.arm_add_f32(np.array([1.,2,3]),np.array([4.,5,7]))

The result of a CMSIS-DSP function will always be a numpy array whatever the arguments were (numpy array or list).

Functions with instance arguments 
*********************************

When the CMSIS-DSP function is requiring an instance data structure, it is just a bit more complex to use it:

First you need to create this instance::

      import cmsisdsp as dsp

      firf32 = dsp.arm_fir_instance_f32()

.. code-block:: C

   void arm_fir_init_f32(
               arm_fir_instance_f32 * S,
               uint16_t numTaps,
         const float32_t * pCoeffs,
               float32_t * pState,
               uint32_t blockSize);


.. py:function:: dsp.arm_fir_init_f32(S,numTaps,pCoeffs,pState)

   Return a list of random ingredients as strings.

   :param S: f32 instance.
   :type S: int
   :param pCoeffs: array.
   :type pCoeffs: NumPy array
   :param pState: array.
   :type pState: NumPy array
   :return: array.
   :rtype: NumPy array

Example of use::

   dsp.arm_fir_init_f32(firf32,3,[1.,2,3],[0,0,0,0,0,0,0])

The third argument in this function is the state. Since all arguments (except the instance ones) are read-only in this Python API, this state will never be changed ! It is just used to communicate the length of the state array which must be allocated by the init function. This argument is required because it is present in the CMSIS-DSP API and in the final C implementation you'll need to allocate a state array with the right dimension.

Since the goal is to be as close as possible to the C API, the API is forcing the use of this argument.

The only change compared to the C API is that the size variables (like blockSize for filter) are computed automatically from the other arguments. This choice was made to make it a bit easier the use of numpy array with the API.

Now, you can check that the instance was initialized correctly::

.. code-block:: python

   print(firf32.numTaps())

The filter can then be called:

.. code-block:: C

   void arm_fir_f32(
               const arm_fir_instance_f32 * S,
               const float32_t * pSrc,
                     float32_t * pDst,
                     uint32_t blockSize);

.. py:function:: dsp.arm_fir_f32(S,pSrc)

   Return a list of random ingredients as strings.

   :param S: f32 instance.
   :type S: int
   :param pSrc: array of input samples.
   :type pSrc: NumPy array
   :return: array.
   :rtype: NumPy array

Then, you can filter with CMSIS-DSP::

   print(dsp.arm_fir_f32(firf32,[1,2,3,4,5]))

The size of this signal should be blockSize. blockSize was inferred from the size of the state array : numTaps + blockSize - 1 according to CMSIS-DSP. So here the signal must have 5 samples.

If you want to filter more than 5 samples, then you can just call the function again. The state variable inside firf32 will ensure that it works like in the CMSIS-DSP C code::

    print(dsp.arm_fir_f32(firf32,[6,7,8,9,10]))

If you want to compare with scipy it is easy but warning : coefficients for the filter are in opposite order in scipy ::

    filtered_x = signal.lfilter([3,2,1.], 1.0, [1,2,3,4,5,6,7,8,9,10])
    print(filtered_x)

FFT
***

The CMSIS-DSP cfft is requiring complex signals with a specific layout in memory.

To remain as close as possible to the C API, we are not using complex numbers in the wrapper. So a complex signal must be converted into a real one. A function like the bewlo one can be used::

   def imToReal1D(a):
       ar=np.zeros(np.array(a.shape) * 2)
       ar[0::2]=a.real
       ar[1::2]=a.imag
       return(ar)

In the same way, the return array from the CMSIS-DSP FFT will not be containing complex Python scalars. It must be converted back with a function like::

   def realToIm1D(ar):
       return(ar[0::2] + 1j * ar[1::2])

Then, the utilisation of the API si very similar to what was done for the FIR example:

Then, you create the FFT instance with::

    cfftf32=dsp.arm_cfft_instance_f32()

You initialize the instance with the init function ::

    status=dsp.arm_cfft_init_f32(cfftf32, nb)
    print(status)

You convert the complex signal to the format expected by the wrapper::

    signalR = imToReal1D(signal)

You compute the FFT of the signal with::

    resultR = dsp.arm_cfft_f32(cfftf32,signalR,0,1)

You convert back to a complex format to compare with scipy::

    resultI = realToIm1D(resultR)
    print(resultI)

Matrix
******

For matrix, the instance variables are masked by the Python API. We decided that for matrix only there was no use for having the CMSIS-DSP instance visibles since they contain the same information as the numpy array (samples and dimension).

So to use a CMSIS-DSP matrix function, it is very simple::

    a=np.array([[1.,2,3,4],[5,6,7,8],[9,10,11,12]])
    b=np.array([[1.,2,3],[5.1,6,7],[9.1,10,11],[5,8,4]])

Numpy result as reference::

    print(np.dot(a , b))

CMSIS-DSP result::

    v=dsp.arm_mat_mult_f32(a,b)
    print(v)

In a real C code, a pointer to a data structure for the result v would have to be passed as argument of the function.

The C API is:

.. code-block:: C

   arm_status arm_mat_mult_f32(
                const arm_matrix_instance_f32 * pSrcA,
                const arm_matrix_instance_f32 * pSrcB,
                      arm_matrix_instance_f32 * pDst);

The Python API is:


.. py:function:: dsp.arm_mat_mult_f32(pSrcA,pSrcB)

   Return the matrix product pSrcA * pSrcB

   :param pSrcA: array of input samples.
   :type pSrcA: NumPy array
   :param pSrcB: array of input samples.
   :type pSrcB: NumPy array
   :return: the matrix product.
   :rtype: NumPy array