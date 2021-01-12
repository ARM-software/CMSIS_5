# Unit tests for CMSIS-NN
Unit test CMSIS-NN functions on any Arm Mbed OS supported HW.

Arm Mbed OS is used for building and flashing.
The Unity test framework is used for running the actual unit tests.

## Requirements

Python3 is required.
It has been tested with Python 3.6 and it has been tested on Ubuntu 16 and 18.

There is a requirement file that can be used to install the dependencies.

```
    ``` pip3 install -r requirements.txt```

```

Note that the exact versions are not required, and there are not a lot of packages to install manually.
The file contains a lot of packages but that is because those are installed when installing some of the other packages.
To manually install packages, see below.

### Executing unit tests

For executing unit tests, the python3 package pyserial is required. Version 3.4 of pyserial has been tested ok.

```
    ``` pip3 install pyserial```

```

Other required python packages are mbed-cli and and mbed-ls. It should not matter if those are installed under python2 or python3 as they are command-line tools. These packages have been tested for Python2, with the following versions: mbed-ls(1.7.9) and mbed-cli(1.10.1).

### Generating new test data

For generating new data, the python3 packages tensorflow, numpy and packaging are required. Tensorflow version 2 is required as a minimum.

## Getting started
Connect any HW (e.g. NUCLEO_F746ZG) that is supported by Arm Mbed OS. Multiple boards are supported. If all requirements are satisfied you can just run:

```
    ```./unittest_targets.py```

```

Use the -h flag to get more info.

It is also possible to build the unit test with Cmake. The binaries can then be used with another test platform, e.g. Fastmodel.
In this case note that toolchain, linker file and Uart code need to be provided. See externs in Retarget.c for specific Uart functions.
UART and LINK_FILE have default values but you most probably need to replace them unless you place your code relatively to the default paths.

```
    ```mkdir build```
    ```cd build```
    ```cmake .. -DCMAKE_TOOLCHAIN_FILE==/path/to/toolchain.cmake -DCPU=cortex-m55 -DUART=/path/to/uart -DLINK_FILE=linkfile```
    ```make```
```

Some examples using Uart and toolchain in Arm Ethos-U Core Software project. See : https://review.mlplatform.org/admin/repos/ml/ethos-u/ethos-u-core-software

```
    ```cmake .. -DCMAKE_TOOLCHAIN_FILE=~/ethos-u-core-software/cmake/toolchain/arm-none-eabi-gcc.cmake -DCMAKE_SYSTEM_PROCESSOR=cortex-m7 -DUART_PATH=~/ethos-u-core-software/drivers/uart -DLINK_FILE=~/platform/fastmodels/model```
    ```cmake .. -DCMAKE_TOOLCHAIN_FILE=~/ethos-u-core-software/cmake/toolchain/arm-none-eabi-gcc.cmake -DCMAKE_SYSTEM_PROCESSOR=cortex-m55 -DUART_PATH=~/ethos-u-core-software/drivers/uart -DLINK_FILE=~/platform/fastmodels/model```
```

If using Cmake it is recommended to use Arm Ethos-U Core Platform project (https://review.mlplatform.org/admin/repos/ml/ethos-u/ethos-u-core-platform) and a fixed virtual platform (FVP) based on Arm Corstone-300 software (https://developer.arm.com/ip-products/subsystem/corstone/corstone-300). First clone the Arm Ethos-U Core Software and Arm Ethos-U Core Platform projects. Also Tensorflow (https://github.com/tensorflow/tensorflow) is needed. It is not used but expected by Arm Ethos-U Core Software. It should be cloned into Arm Ethos-U Core Software. Then build:

```
    ```mkdir build```
    ```cd build```
    ```cmake .. -DCMAKE_TOOLCHAIN_FILE=</path/to/Ethos-u-core-platform>/cmake/toolchain/arm-none-eabi-gcc.cmake -DETHOSU_CORE_PLATFORM_PATH=</path/to/Ethos-u-core-platform> -DUSE_ETHOSU_CORE_PLATFORM=ON -DTARGET_CPU=cortex-m55 -DETHOS_U_CORE_SOFTWARE_PATH=</path/to/Ethos-u-core-software> -DCORE_SOFTWARE_ACCELERATOR=CMSIS-NN -DCORE_SOFTWARE_RTOS=None```
    ```make test_arm_depthwise_conv_s8_opt```
```
Note that here you may want to specifiy the unit test targets to build otherwise it will build external targets as well. Some more examples, assuming Ethos-u-core-platform and Ethos-u-core_software are cloned into your home directory:

```
    ```cmake .. -DCMAKE_TOOLCHAIN_FILE=~/ethos-u-core-platform/cmake/toolchain/arm-none-eabi-gcc.cmake -DETHOSU_CORE_PLATFORM_PATH=~/ethos-u-core-platform -DUSE_ETHOSU_CORE_PLATFORM=ON -DTARGET_CPU=cortex-m55 -DETHOS_U_CORE_SOFTWARE_PATH=~/ethos-u-core-software -DCORE_SOFTWARE_ACCELERATOR=CMSIS-NN -DCORE_SOFTWARE_RTOS=None```
    ```cmake .. -DCMAKE_TOOLCHAIN_FILE=~/ethos-u-core-platform/cmake/toolchain/arm-none-eabi-gcc.cmake -DETHOSU_CORE_PLATFORM_PATH=~/ethos-u-core-platform -DUSE_ETHOSU_CORE_PLATFORM=ON -DTARGET_CPU=cortex-m7 -DETHOS_U_CORE_SOFTWARE_PATH=~/ethos-u-core-software -DCORE_SOFTWARE_ACCELERATOR=CMSIS-NN -DCORE_SOFTWARE_RTOS=None```
    ```cmake .. -DCMAKE_TOOLCHAIN_FILE=~/ethos-u-core-platform/cmake/toolchain/armclang.cmake -DETHOSU_CORE_PLATFORM_PATH=~/ethos-u-core-platform -DUSE_ETHOSU_CORE_PLATFORM=ON -DTARGET_CPU=cortex-m3 -DETHOS_U_CORE_SOFTWARE_PATH=~/ethos-u-core-software -DCORE_SOFTWARE_ACCELERATOR=CMSIS-NN -DCORE_SOFTWARE_RTOS=None```
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
    ```./generate_test_data.py -h```

```

The script use a concept of test data sets, i.e. it need a test set data name as input. It will then generate files with that name as prefix. Multiple header files of different test sets can then be included in the actual unit test files.
When adding a new test data set, new c files should be added or existing c files should be updated to use the new data set. See overview of the folders on how/where to add new c files.

As it is now, when adding a new test data set, you would first have to go and edit the script to configure the parameters as you want.
Once you are happy with the new test data set, it should be added in the load_all_testdatasets() function.

## Overview of the Folders

- `Output` - This will be created when building.
- `Mbed` - These are the Arm Mbed OS settings that are used. See Mbed/README.md.
- `PregeneratedData` - These are tests sets of data that have been previously been generated and are used in the unit tests.
- `TestCases` - Here are the actual unit tests. For each function under test there is a folder under here.
- `TestCases/<cmsis-nn function name>` - For each function under test there is a folder with the same name with test_ prepended to the name and it contains a c-file with the actual unit tests. For example for arm_convolve_s8() the file is called test_arm_convolve_s8.c
- `TestCases/<cmsis-nn function name>/Unity` - This folder contains a Unity file that calls the actual unit tests. For example for arm_convolve_s8() the file is called unity_test_arm_convolve_s8.c.
- `TestCases/<cmsis-nn function name>/Unity/TestRunner` - This folder will contain the autogenerated Unity test runner.
- `TestCases/TestData` - This is auto generated test data that the unit tests are using. It is the same data as in the PregenrateData folder but in actual C header format. The advantage of having the same data in two places, is that the data can be easily regenerated (randomized) with the same config. All data can regenerated or only parts of it (e.g. only bias data). Of course even the config can be regenerated. This might be useful during debugging.
- `Platform` - This is only used when using Cmake. It is handling dependencies lika Uart, that would otherwise be handled by Arm Mbed OS.
