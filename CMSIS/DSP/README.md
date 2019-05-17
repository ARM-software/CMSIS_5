# README

## How to use

This document is explaining how to use cmake with CMSIS-DSP and AC6 ARM compiler.

The examples arm_variance_f32 in folder Examples/ARM/arm_variance_f32 has been modified to also
support cmake and is used as an example in this document.

If you don't use AC6, you'll need to modify the cmake files as explained below.

### Generating the Makefiles

To build example arm_variance_f32 with cmake, you need to create a build folder where the build will take place. Don't build in your source directory.

You can create a build folder in Examples/ARM/arm_variance_f32

Once you are in the build folder, you can use cmake to generate the Makefiles.

For instance, to build for m7 :
cmake -DCMAKE_TOOLCHAIN_FILE=../../../../armcc.cmake -DARM_CPU="cortex-m7" -G "Unix Makefiles" ..

To build for A5
cmake -DCMAKE_TOOLCHAIN_FILE=../../../../armcc.cmake -DARM_CPU="cortex-a5" -G "Unix Makefiles" ..

To build for A5 with Neon acceleration
cmake -DCMAKE_TOOLCHAIN_FILE=../../../../armcc.cmake -DNEON=ON -DARM_CPU="cortex-a5" -G "Unix Makefiles" ..

cmake will check it can find the cross compiling tools as defined in armcc.cmake

### Toolchain 

You may have to change the "tools" variable in armcc.make. It is pointing to your toolchain.
The version of armcc.cmake on github is using the ARM AC6 compiler coming from the ArmDS environment.  The tools variable is thus pointing to ArmDS.

If you use a different clang toolchain, you can just modify the tools path.

If you build with gcc, you'll need to change armcc.cmake, config.cmake and configUtils.cmake

config.make is defining options like -mfpu and the value to pass to gcc (or other compiler) may be different.

configUtils.cmake is defining the use of a scatter file and it may be different with gcc.

### Building 

make VERBOSE=1

### Running

The executable can run on a FVP. 
For instance, if you built for m7, you could just do:

FVP_MPS2_Cortex-M7.exe -a arm_variance_example

## Customization

armcc.make is use to cross compil with AC6 coming from ArmDS.

You'll need to create a different toolchain file if you use something different.
Then you'll need to pass this file to cmake on the command line.

config.cmake is included by the CMSIS-DSP cmake and is defining the options and include paths
needed to compile CMSIS-DSP.

configBoot.cmake are definitions required to run an executable on a platform. It is using files from the Device folder of CMSIS. The result can run on FVP.

If you need to run on something different, you'll need to modfy configBoot. If you need a different scatter file you'll need to modify configBoot.

configBoot is relying on some functions defined in configUtils and most of the customizations should be done here.


