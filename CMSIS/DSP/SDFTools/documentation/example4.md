# Example 4

It is exactly the same example as example 3 but the code generation is generating Python code instead of C++.

The Python code is generated with:

```python
sched.pythoncode(".",config=conf)
```

and it will generate a `sched.py` file.

A file `custom.py` and `appnodes.py` are also required.

## custom.py

```python
import numpy as np 

HANN=np.array([0.,... ])

```

An array HANN is defined for the Hann window.

## appnodes.py

This file is defining the new nodes which were used in `graph.py`. In `graph.py` which are just defining new kind of nodes for scheduling purpose : type and sizes.

In `appnodes.py` we including new kind of nodes for simulation purpose:

```python
from sdf.nodes.py.CFFTF32 import *
```



The CFFT is vey similar to the C++ version of example 3:

```python
class CFFT(GenericNode):
    def __init__(self,inputSize,outSize,fifoin,fifoout):
        GenericNode.__init__(self,inputSize,outSize,fifoin,fifoout)
        self._cfftf32=dsp.arm_cfft_instance_f32()
        status=dsp.arm_cfft_init_f32(self._cfftf32,inputSize>>1)

    def run(self):
        a=self.getReadBuffer()
        b=self.getWriteBuffer()
        # Copy arrays (not just assign references)
        b[:]=a[:]
        dsp.arm_cfft_f32(self._cfftf32,b,0,1)
        return(0)
```

The line `b[:] = a[:]` is like the memcpy of the C++ version.



It is important when using Numpy to do something like:

```python
b[:] = ...
```

Because we want to write into the write buffer.

If we were writing:

```python
b=a
# OR
b=a.copy()
```

we would just be assigning a new reference to variable `b` and discard the previous `b` buffer. It would not work. When writing new nodes, it must be kept in mind.

In this example we also want to display the output with matplotlib.

The Python FileSink is taking another argument : the matplotlib buffer. So, it is a little bit different from the C++ version since we also need to pass this new argument to the node.

So, in graph.py we have:



```python
sink=FileSink("sink",AUDIO_INTERRUPT_LENGTH)
sink.addLiteralArg("output_example3.txt")
sink.addVariableArg("dispbuf")
```

Then, in the configuration object we define an argument for the scheduling function:



```python
conf=Configuration()
conf.pyOptionalArgs="dispbuf"
```



And, in our main.py we pass a buffer to the scheduling function:



```python
DISPBUF = np.zeros(16000)

nb,error = s.scheduler(DISPBUF)
```

