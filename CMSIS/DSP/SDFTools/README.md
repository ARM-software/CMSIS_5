# Synchronous Dataflow for CMSIS-DSP

## Introduction

A dataflow graph is a representation of how compute blocks are connected to implement a processing. 

Here is an example with 3 nodes:

- A source
- A filter
- A sink

Each node is producing and consuming some amount of samples. For instance, the source node is producing 5 samples each time it is run. The filter node is consuming 7 samples each time it is run.



The FIFOs lengths are represented on each edge of the graph : 11 samples for the leftmost FIFO and 5 for the other one.

In blue, the amount of samples generated or consumed by a node each time it is called.

<img src="documentation/graph1.PNG" alt="graph1" style="zoom:50%;" />

When the processing is applied to a stream of samples then the problem to solve is : 

**how the blocks must be scheduled and the FIFOs connecting the block dimensioned**

The general problem can be very difficult. But, if some constraints are applied to the graph then some algorithms can compute a static schedule.

When the following constraints are satisfied we say we have a Synchronous Dataflow Graph (SDF):

- Static graph : graph topology is not changing
- Each node is always consuming and producing the same number of samples

The CMSIS-DSP SDF Tools are a set of Python scripts and C++ classes with following features:

- A SDF can be described in Python
- The Python script will compute a static schedule and the FIFOs size
- A static schedule is:
  - A periodic sequence of functions calls
  - A periodic execution where the FIFOs remain bounded
  - A periodic execution with no deadlock : when a node is run there is enough data available to run it 
- The Python script will generate a [Graphviz](https://graphviz.org/) representation of the graph 
- The Python script will generate a C++ implementation of the static schedule 
- The Python script can also generate a Python implementation of the static schedule (for use with the CMSIS-DSP Python wrapper)



## Why it is useful

Without any scheduling tool for a dataflow graph, there is a problem of modularity : a change on a node may impact other nodes in the graph. For instance, if the number of samples consumed by a node is changed:

- You may need to change how many samples are produced by the predecessor blocks  in the graph (assuming it is possible)
- You may need to change how many times the predecessor blocks must run
- You may have to change the FIFOs sizes

With the CMSIS-DSP SDF Tools you don't have to think about those details while you are still experimenting with your data processing pipeline. It makes it easier to experiment, add or remove blocks, change their parameters.

The tools will generate a schedule and the FIFOs. Even if you don't use this at the end for a final implementation, the information could be useful : is the schedule too long ? Are the FIFOs too big ?

Let's look at an (artificial) example:

<img src="documentation/graph1.PNG" alt="graph1" style="zoom:50%;" />

Without a tool, the user would probably try to modify the sample values so that the number of sample produced is equal to the number of samples consumed. With the SDF Tools  we know that such a graph can be scheduled and that the FIFO sizes need to be 11 and 5.

The periodic schedule generated for this graph has a length of 19. It is big for such a small graph and it is because, indeed 5 and 7 are not very well chosen values. But, it is working even with those values.

The schedule is (the size of the FIFOs after the execution of the node displayed in the brackets):

`source [ 5   0]`
`source [10   0]`
`filter [ 3   5]`
`sink   [ 3   0]`
`source [ 8   0]`
`filter [ 1   5]`
`sink   [ 1   0]`
`source [ 6   0]`
`source [11   0]`
`filter [ 4   5]`
`sink   [ 4   0]`
`source [ 9   0]`
`filter [ 2   5]`
`sink   [ 2   0]`
`source [ 7   0]`
`filter [ 0   5]`
`sink   [ 0   0]`

At the end, both FIFOs are empty so the schedule can be run again : it is periodic !

## How to build the examples

First, you must install the `CMSIS-DSP` PythonWrapper:

```
pip install cmsisdsp
```

In folder `SDFTools/example/build`, type the `cmake` command:

```bash
cmake -DHOST=YES -DDOT="path to dot tool" -DCMSIS="path to cmsis" -G "Unix Makefiles" ..
```

The Graphviz dot tool is requiring a recent version supporting the HTML-like labels.

The path to cmsis must be the root folder containing CMSIS and Device folders.

If cmake is successful, you can type `make` to build the examples. It will also build CMSIS-DSP for the host.

If you don't have graphviz, the option -DDOT can be removed.

If for some reason it does not work, you can go into an example folder (for instance example1), and type the commands:

```bash
python graph.py 
dot -Tpdf -o test.pdf test.dot
```

It will generate the C++ files for the schedule and a pdf representation of the graph.

Note that the Python code is relying on the CMSIS-DSP PythonWrapper which is now also containing the Python scripts for the Synchronous Data Flow.

To build the C examples:

* CMSIS-DSP must be built, 
* the .cpp file contained in the example must be built
* the include folder `sdf/src` must be added

For `example3` which is using an input file, cmake should have copied the input test pattern `input_example3.txt` inside the build folder. The output file will also be generated in the build folder.

`example4` is like `example3` but in pure Python and using the CMSIS-DSP Python wrapper (which must already be installed before trying the example). `example4` is not built by the cmake. You'll need to go to the `example4` folder and type:

```bash
python graph.py 
python main.py
```

The first line is generating the schedule in Python. The second line is executing the schedule.

`example7` is communicating with `OpenModelica`. You need to install the VHTModelica blocks from the [VHT-SystemModeling](https://github.com/ARM-software/VHT-SystemModeling) project on our GitHub

## Limitations

It is a first version and there are lots of limitations and probably bugs:

- The code generation is using [Jinja](https://jinja.palletsprojects.com/en/3.0.x/) template in `sdf/templates`. They must be cleaned to be more readable. You can modify the templates according to your needs ;
- CMSIS-DSP integration must be improved to make it easier
- Some optimizations are missing 
- Some checks are missing : for instance you can connect several nodes to the same io port. And io port must be connected to only one other io port. It is not checked by the script.
- The code is requiring a lot more comments and cleaning
- A C version of the code generator is missing
- The scheduling algorithm could provide different heuristics:
  - Optimizing code size
  - Optimizing memory usage 
- The code generation could provide more flexibility for memory allocation with a choice between:
  - Global
  - Stack
  - Heap

## Default nodes
Here is a list of the nodes supported by default. More can be easily added:

- Unary:
  - Unary function with header `void function(T* src, T* dst, int nbSamples)`
- Binary:
  - Binary function with header `void function(T* srcA, T* srcB, T* dst, int nbSamples)`
- CMSIS-DSP function:
  - It will detect if it is an unary or binary function.
  - The name must not contain the prefix arm nor the the type suffix
  - For instance, use `Dsp("mult",CType(F32),NBSAMPLES)` to use `arm_mult_f32`
  - Other CMSIS-DSP function (with an instance variable) are requiring the creation of a Node if it is not already provided
- CFFT / ICFFT : Use of CMSIS-DSP CFFT. Currently only F32 and Q15 
- Zip / Unzip : To zip / unzip streams 
- ToComplex : Map a real stream onto a complex stream
- ToReal : Extract real part of a complex stream
- FileSource and FileSink : Read/write float to/from a file
- NullSink : Do nothing. Useful for debug 
- StereoToMonoQ15 : Interleaved stereo converted to mono with scaling to avoid saturation of the addition
- Python only nodes:
  - WavSink and WavSource to use wav files for testing
  - VHTSDF : To communicate with OpenModelica using VHTModelica blocks


## Detailed examples 

- [Example 1 : how to describe a simple graph](documentation/example1.md)
- [Example 2 : More complex example with delay and CMSIS-DSP](documentation/example2.md)
- [Example 3 : Working example with CMSIS-DSP and FFT](documentation/example3.md)
- [Example 4 : Same as example 3 but with the CMSIS-DSP Python wrapper](documentation/example4.md)

Examples 5 and 6 are showing how to use the CMSIS-DSP MFCC with a synchronous data flow.

Example 7 is communicating with OpenModelica. The Modelica model (PythonTest) in the example is implementing a Larsen effect.
