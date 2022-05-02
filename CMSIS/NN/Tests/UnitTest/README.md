# Unit tests for CMSIS-NN

Unit test CMSIS-NN functions on any [Arm Mbed OS](https://os.mbed.com/mbed-os/) supported HW or using a fixed virtual platform (FVP) based on [Arm Corstone-300 software](https://developer.arm.com/ip-products/subsystem/corstone/corstone-300).

The [Unity test framework](http://www.throwtheswitch.org/unity) is used for running the actual unit tests.

## Requirements

Python3 is required.
It has been tested with Python 3.6 and it has been tested on Ubuntu 16 and 18.

Make sure to use a `pip` version > 19.0 (or >20.3 for macOS), otherwise tensorflow 2 packages are not available.
If in a virtual environment just start by upgrading pip.

```
pip install --upgrade pip
```

Note that the exact versions are not required, and there are not a lot of packages to install manually.
The file contains a lot of packages but that is because those are installed when installing some of the other packages.
To manually install packages, see below.

### Executing unit tests

If using the script unittest_targets.py for executing unit tests, the following packages are needed.

```
pip install pyserial mbed-ls termcolor
```

Other required python packages are mbed-cli and and mbed-ls. It should not matter if those are installed under python2 or python3 as they are command-line tools. These packages have been tested for Python2, with the following versions: mbed-ls(1.7.9) and mbed-cli(1.10.1).

### Generating new test data

For generating new test data, the following packages are needed.

```
pip install numpy packaging tensorflow
```


For generating new data, the python3 packages tensorflow, numpy and packaging are required. Most unit tests use a Keras generated model for reference. The SVDF unit test use a json template as input for generating a model. To do so flatc compiler is needed and it requires a schema file.

#### Get flatc and schema

Note this is only needed for generating SVDF unit tests.

For flatc compiler clone this [repo](https://github.com/google/flatbuffers) and build:
```
cd flatbuffers
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
make
```
Remember to add the built flatc binary to the path.

For schema file download [schema.fbs](https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/schema/schema.fbs).

## Getting started

### Using Arm Mbed OS supported hardware

Connect any HW (e.g. NUCLEO_F746ZG) that is supported by Arm Mbed OS. Multiple boards are supported. If all requirements are satisfied you can just run:

```
./unittest_targets.py
```

Use the -h flag to get more info.

### Using FVP based on Arm Corstone-300 software

It is recommended to use toolchain files from [Arm Ethos-U Core Platform](https://review.mlplatform.org/admin/repos/ml/ethos-u/ethos-u-core-platform) project. These are supporting TARGET_CPU, which is a required argument. Note that if not specifying TARGET_CPU, these toolchains will set some default. The format must be TARGET_CPU=cortex-mXX, see examples below.
Clone Arm Ethos-U Core Platform project and build:

```
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=</path/to/ethos-u-core-platform>/cmake/toolchain/arm-none-eabi-gcc.cmake -DTARGET_CPU=cortex-m55
make
```

This will build all unit tests. You can also just build a specific unit test only, for example:

```
make test_arm_depthwise_conv_s8_opt
```

Some more examples, assuming Ethos-u-core-platform is cloned into your home directory:

```
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/ethos-u-core-platform/cmake/toolchain/arm-none-eabi-gcc.cmake -DTARGET_CPU=cortex-m55
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/ethos-u-core-platform/cmake/toolchain/arm-none-eabi-gcc.cmake -DTARGET_CPU=cortex-m7
cmake .. -DCMAKE_TOOLCHAIN_FILE=~/ethos-u-core-platform/cmake/toolchain/armclang.cmake -DTARGET_CPU=cortex-m3
```

Then you need to download and install the FVP based Arm Corstone-300 software, for example:

```
mkdir -p /home/$user/FVP
wget https://developer.arm.com/-/media/Arm%20Developer%20Community/Downloads/OSS/FVP/Corstone-300/FVP_Corstone_SSE-300_Ethos-U55_11.12_57.tgz
tar -xvzf FVP_Corstone_SSE-300_Ethos-U55_11.12_57.tgz
./FVP_Corstone_SSE-300_Ethos-U55.sh --i-agree-to-the-contained-eula --no-interactive -d /home/$user/FVP
export PATH="/home/$user/FVP/models/Linux64_GCC-6.4:$PATH"
```

Finally you can run the unit tests. For example:

```
FVP_Corstone_SSE-300_Ethos-U55 --cpulimit 2 -C mps3_board.visualisation.disable-visualisation=1 -C mps3_board.telnetterminal0.start_telnet=0 -C mps3_board.uart0.out_file="-" -C mps3_board.uart0.unbuffered_output=1 ./TestCases/test_arm_depthwise_conv_s8_opt/test_arm_depthwise_conv_s8_opt.elf
```

## Generating new test data

Generating new test data is done with the following script. Use the -h flag to get more info.

```
./generate_test_data.py -h

```

The script use a concept of test data sets, i.e. it need a test set data name as input. It will then generate files with that name as prefix. Multiple header files of different test sets can then be included in the actual unit test files.
When adding a new test data set, new c files should be added or existing c files should be updated to use the new data set. See overview of the folders on how/where to add new c files.

As it is now, when adding a new test data set, you would first have to go and edit the script to configure the parameters as you want.
Once you are happy with the new test data set, it should be added in the load_all_testdatasets() function.

## Overview of the Folders

- `Corstone-300` - These are dependencies, like linker files etc, needed when building binaries targetting the FVP based on Arm Corstone-300 software. This is mostly taken from Arm Ethos-U Core Platform project.
- `Mbed` - These are the Arm Mbed OS settings that are used. See Mbed/README.md.
- `Output` - This will be created when building.
- `PregeneratedData` - These are tests sets of data that have been previously been generated and are used in the unit tests.
- `TestCases` - Here are the actual unit tests. For each function under test there is a folder under here.
- `TestCases/<cmsis-nn function name>` - For each function under test there is a folder with the same name with test_ prepended to the name and it contains a c-file with the actual unit tests. For example for arm_convolve_s8() the file is called test_arm_convolve_s8.c
- `TestCases/<cmsis-nn function name>/Unity` - This folder contains a Unity file that calls the actual unit tests. For example for arm_convolve_s8() the file is called unity_test_arm_convolve_s8.c.
- `TestCases/<cmsis-nn function name>/Unity/TestRunner` - This folder will contain the autogenerated Unity test runner.
- `TestCases/TestData` - This is auto generated test data that the unit tests are using. It is the same data as in the PregenrateData folder but in actual C header format. The advantage of having the same data in two places, is that the data can be easily regenerated (randomized) with the same config. All data can regenerated or only parts of it (e.g. only bias data). Of course even the config can be regenerated. This might be useful during debugging.
- `TestCases/Common` - Common files used in test data generation is placed here.
