# Setting Linux is forcing th extension to be .o instead of .obj when building on WIndows.
# It is important because armlink is failing when files have .obj extensions (error with
# scatter file section not found)
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR arm)




#SET(CMAKE_C_COMPILER "${tools}/bin/arm-none-eabi-gcc")
#SET(CMAKE_CXX_COMPILER "${tools}/bin/arm-none-eabi-g++")
#SET(CMAKE_ASM_COMPILER "${tools}/bin/arm-none-eabi-gcc")

find_program(CMAKE_C_COMPILER NAMES arm-none-eabi-gcc arm-none-eabi-gcc.exe)
find_program(CMAKE_CXX_COMPILER NAMES arm-none-eabi-g++ arm-none-eabi-g++.exe)
find_program(CMAKE_ASM_COMPILER NAMES arm-none-eabi-gcc arm-none-eabi-gcc.exe)


if (NOT ("${tools}" STREQUAL ""))
message(STATUS "Tools path is set")
SET(CMAKE_AR "${tools}/bin/ar")
SET(CMAKE_CXX_COMPILER_AR "${tools}/bin/ar")
SET(CMAKE_C_COMPILER_AR "${tools}/bin/ar")
else()
find_program(CMAKE_AR NAMES arm-none-eabi-gcc-ar arm-none-eabi-gcc-ar.exe )
find_program(CMAKE_CXX_COMPILER_AR NAMES arm-none-eabi-gcc-ar arm-none-eabi-gcc-ar.exe )
find_program(CMAKE_C_COMPILER_AR NAMES arm-none-eabi-gcc-ar arm-none-eabi-gcc-ar.exe)
endif()

#SET(CMAKE_LINKER "${tools}/bin/arm-none-eabi-g++")
find_program(CMAKE_LINKER NAMES arm-none-eabi-g++ arm-none-eabi-g++.exe)

SET(CMAKE_C_LINK_EXECUTABLE "<CMAKE_LINKER> <LINK_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
SET(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_LINKER> <LINK_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
SET(CMAKE_C_OUTPUT_EXTENSION .o)
SET(CMAKE_CXX_OUTPUT_EXTENSION .o)
SET(CMAKE_ASM_OUTPUT_EXTENSION .o)
# When library defined as STATIC, this line is needed to describe how the .a file must be
# create. Some changes to the line may be needed.
SET(CMAKE_C_CREATE_STATIC_LIBRARY "<CMAKE_AR> -crs <TARGET> <LINK_FLAGS> <OBJECTS>" )
SET(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> -crs <TARGET> <LINK_FLAGS> <OBJECTS>" )

set(GCC ON)
# default core

if(NOT ARM_CPU)
    set(
            ARM_CPU "cortex-a5"
            CACHE STRING "Set ARM CPU. Default : cortex-a5"
    )
endif(NOT ARM_CPU)

if (ARM_CPU STREQUAL "cortex-m55")
# For gcc 10q4
SET(CMAKE_C_FLAGS "-ffunction-sections -fdata-sections -march=armv8.1-m.main+mve.fp+fp.dp" CACHE INTERNAL "C compiler common flags")
SET(CMAKE_CXX_FLAGS "-ffunction-sections -fdata-sections -march=armv8.1-m.main+mve.fp+fp.dp" CACHE INTERNAL "C compiler common flags")
SET(CMAKE_ASM_FLAGS "-march=armv8.1-m.main+mve.fp+fp.dp" CACHE INTERNAL "ASM compiler common flags")
SET(CMAKE_EXE_LINKER_FLAGS "-fno-use-linker-plugin -march=armv8.1-m.main+mve.fp+fp.dp"  CACHE INTERNAL "linker flags")
elseif (ARM_CPU STREQUAL "cortex-m55+nomve.fp+nofp")
# This case is not tested nor supported
SET(CMAKE_C_FLAGS "-ffunction-sections -fdata-sections -march=armv8.1-m.main+dsp+fp.dp" CACHE INTERNAL "C compiler common flags")
SET(CMAKE_CXX_FLAGS "-ffunction-sections -fdata-sections -march=armv8.1-m.main+dsp+fp.dp" CACHE INTERNAL "C compiler common flags")
SET(CMAKE_ASM_FLAGS "-march=armv8.1-m.main+dsp+fp.dp" CACHE INTERNAL "ASM compiler common flags")
SET(CMAKE_EXE_LINKER_FLAGS "-fno-use-linker-plugin -march=armv8.1-m.main+dsp+fp.dp"  CACHE INTERNAL "linker flags")
else()
SET(CMAKE_C_FLAGS "-ffunction-sections -fdata-sections -mcpu=${ARM_CPU}" CACHE INTERNAL "C compiler common flags")
SET(CMAKE_CXX_FLAGS "-ffunction-sections -fdata-sections -mcpu=${ARM_CPU}" CACHE INTERNAL "C compiler common flags")
SET(CMAKE_ASM_FLAGS "-mcpu=${ARM_CPU}" CACHE INTERNAL "ASM compiler common flags")
SET(CMAKE_EXE_LINKER_FLAGS "-mcpu=${ARM_CPU}"  CACHE INTERNAL "linker flags")
endif()

get_property(IS_IN_TRY_COMPILE GLOBAL PROPERTY IN_TRY_COMPILE)
if(IS_IN_TRY_COMPILE)
    add_link_options("--specs=nosys.specs")
endif()

add_link_options("-Wl,--start-group")
#add_link_options("-mcpu=${ARM_CPU}")

# Where is the target environment
#SET(CMAKE_FIND_ROOT_PATH "${tools}")
# Search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# For libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

