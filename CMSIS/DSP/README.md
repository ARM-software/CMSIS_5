# README

## Folders and files

### Folders

* cmsisdsp
  * Require to build the CMSIS-DSP PythonWrapper for the Python repository
  * It contains all Python packages
* ComputeLibrary:
  * Some kernels required when building CMSIS-DSP with Neon acceleration
* Examples:
  * Examples of use of CMSIS-DSP
* Include:
  * Include files for CMSIS-DSP
* Platforms:
  * Used for running the CMSIS-DSP test framework on Arm IPSS
* PrivateInclude:
  * Some include needed to build CMSIS-DSP
* PythonWrapper:
  * C code for the CMSIS-DSP PythonWrapper
  * Examples for the PythonWrapper
* Scripts:
  * Debugging scripts
  * Script to generate some coefficient tables used by CMSIS-DSP
* SDFTools:
  * Examples for the Synchronous Data Flow
  * C++ templates for the Synchronous Data Flow
* Source:
  * CMSIS-DSP source
* Testing:
  * CMSIS-DSP Test frameworks
* Toolchain:
  * cmake files for building with gcc or Arm clang

### Files

Some files are needed to generate the PythonWrapper:

* PythonWrapper_README.md
* LICENSE.txt
* MANIFEST.in
* pyproject.toml
* setup.py

cmake files are useful to build CMSIS-DSP and the Test framework.

filterLinkScript.py is used by the cmake build.

cmsisdspconfig.py:

* Web browser UI to generate build configurations



## How to use

This document is explaining how to use cmake with CMSIS-DSP.
(It is not official so not supported. The official way to build is to use the CMSIS-Pack).

The example arm_variance_f32 in folder Examples/ARM/arm_variance_f32 has been modified to also
support cmake and is used as an example in this document.

### Generating the Makefiles

To build example arm_variance_f32 with cmake, you need to create a build folder where the build will take place. Don't build in your source directory.

You can create a build folder in Examples/ARM/arm_variance_f32

Once you are in the build folder, you can use cmake to generate the Makefiles.

The cmake command is requiring several arguments. For instance, to build for m7 with AC6 compiler:

    cmake -DCMAKE_PREFIX_PATH="path to compiler (folder containing the bin folder)" \
    -DCMAKE_TOOLCHAIN_FILE="../../../../armac6.cmake" \
    -DARM_CPU="cortex-m7" \
    -DROOT="../../../../../.." \
    -DPLATFORM="FVP" \
    -G "Unix Makefiles" ..

DCMAKE_PREFIX_PATH is the path to the compiler toolchain. This folder should contain the bin folder where are the compiler executables.

ROOT is pointing to the root CMSIS folder (the one containing CMSIS and Device).

PLATFORM is used to select the boot code for the example. In example below, Fast Model (FVP) is selected and the boot code for fast model will be used.

CMAKE_TOOLCHAIN_FILE is selecting the toolchain (ac6, ac5 or gcc). The files are in CMSIS/DSP.

ARM_CPU is selecting the core. The syntax must be the one recognized by the compiler.
(So for instance different between AC5 and AC6).

The final .. is the path to the directory containing the CMakeLists.txt of the variance example.
Since the build folder is assumed to be created in arm_variance_examples then the path to CMakeLists.txt from the build folder is ..

To build for A5, you need to change DCMAKE_TOOLCHAIN_FILE and ARM_CPU:

    -DCMAKE_TOOLCHAIN_FILE=../../../../armac5.cmake 
    -DARM_CPU="cortex-a5"

To build for A5 with Neon acceleration, you need to add:

    -DNEON=ON

### Building 

Once cmake has generated the makefiles, you can use a GNU Make to build.

    make VERBOSE=1

### Running

The generated executable can be run on a fast model. 
For instance, if you built for m7, you could just do:

    FVP_MPS2_Cortex-M7.exe -a arm_variance_example

The final executable has no extension in the filename. 

## Building only the CMSIS-DSP library

If you want to build only the CMSIS-DSP library and don't link with any boot code then you'll need to write a specific cmake.

Create a folder BuildCMSISOnly.

Inside the folder, create a CMakeLists.txt with the following content:

```cmake
cmake_minimum_required (VERSION 3.14)

# Define the project
project (testcmsisdsp VERSION 0.1)

# Define the path to CMSIS-DSP (ROOT is defined on command line when using cmake)
set(DSP ${ROOT}/CMSIS/DSP)

# Add DSP folder to module path
list(APPEND CMAKE_MODULE_PATH ${DSP})

########### 
#
# CMSIS DSP
#

# Load CMSIS-DSP definitions. Libraries will be built in bin_dsp
add_subdirectory(${DSP}/Source bin_dsp)
```

Create a build folder inside the BuildCMSISOnly folder.

Inside the build folder, type following cmake command

    cmake -DROOT="path to CMSIS Root" \
      -DCMAKE_PREFIX_PATH="path to compiler (folder containing the bin folder)" \
      -DCMAKE_TOOLCHAIN_FILE="../../CMSIS_ROOT/CMSIS/DSP/armac6.cmake" \
      -DARM_CPU="cortex-m7" \
      -G "Unix Makefiles" ..

Now you can make:

    make VERBOSE=1

When the build has finished, you'll have a bin_dsp folder inside your build folder.
Inside bin_dsp, you'll have a folder per CMSIS-DSP Folder : BasicMathFunctions ...

Inside each of those folders, you'll  have a library : libCMSISDSPBasicMath.a ...



## Compilation symbols for tables

Some new compilations symbols have been introduced to avoid including all the tables if they are not needed.

If no new symbol is defined, everything will behave as usual. If ARM_DSP_CONFIG_TABLES is defined then the new symbols will be taken into account.

It is strongly suggested to use the new Python script cmsisdspconfig.py to generate the -D options to use on the compiler command line.

    pip install streamlit
    streamlit run cmsisdspconfig.py

If you use cmake, it is also easy since high level options are defined and they will select the right compilation symbols. 

For instance, if you want to use the arm_rfft_fast_f32, in fft.cmake you'll see an option RFFT_FAST_F32_32.

If you don't use cmake nor the Python script, you can just look at fft.cmake or interpol.cmake in Source to see which compilation symbols are needed.

We see, for arm_rfft_fast_f32, that the following symbols need to be enabled :

* ARM_TABLE_TWIDDLECOEF_F32_16 
* ARM_TABLE_BITREVIDX_FLT_16
* ARM_TABLE_TWIDDLECOEF_RFFT_F32_32
* ARM_TABLE_TWIDDLECOEF_F32_16

In addition to that, ARM_DSP_CONFIG_TABLES must be enabled and finally ARM_FFT_ALLOW_TABLES must also be defined.

This last symbol is required because if no transform functions are included in the build, then by default all flags related to FFT tables are ignored.


## Bit Reverse Tables CMSIS DSP

It is a question coming often.

It is now detailed [in this github issue](https://github.com/ARM-software/CMSIS_5/issues/858)

Someone from the community has written a [Python script to help](https://gist.github.com/rosek86/d0d709852fddf36193071d7f61987bae)