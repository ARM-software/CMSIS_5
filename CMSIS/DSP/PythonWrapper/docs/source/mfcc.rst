MFCC
====

.. highlight:: python

.. autofunction:: cmsisdsp.mfcc.frequencyToMelSpace
.. autofunction:: cmsisdsp.mfcc.melSpaceToFrequency

MEL filters are represented as 3 arrays to encode a sparse matrix.

.. autofunction:: cmsisdsp.mfcc.melFilterMatrix

The API is returning a tuple::
  
  (filterLength,filterPos,PackedCoefficients)

The first argument is the list of MEL filter lengths.

The second argument is the position of the first coefficient of the filter in the list of coefficients,

The last argument is the concatenated list of all filter coefficients.



.. autofunction:: cmsisdsp.mfcc.dctMatrix

