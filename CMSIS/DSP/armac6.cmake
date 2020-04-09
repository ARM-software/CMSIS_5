# Setting Linux is forcing th extension to be .o instead of .obj when building on WIndows.
# It is important because armlink is failing when files have .obj extensions (error with
# scatter file section not found)
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR arm)



#SET(tools "C:/PROGRA~1/ARM/DEVELO~1.0/sw/ARMCOM~1.12")

find_program(CMAKE_C_COMPILER NAMES armclang armclang.exe)
find_program(CMAKE_CXX_COMPILER NAMES armclang armclang.exe)
find_program(CMAKE_ASM_COMPILER NAMES armclang armclang.exe)

#SET(CMAKE_C_COMPILER "${tools}/bin/armclang.exe")
#SET(CMAKE_CXX_COMPILER "${tools}/bin/armclang.exe")
#SET(CMAKE_ASM_COMPILER "${tools}/bin/armclang.exe")

find_program(CMAKE_AR NAMES armar armar.exe)

#SET(CMAKE_AR "${tools}/bin/armar.exe")
find_program(CMAKE_CXX_COMPILER_AR NAMES armar armar.exe)
find_program(CMAKE_C_COMPILER_AR NAMES armar armar.exe)

find_program(CMAKE_LINKER NAMES armlink armlink.exe)
#SET(CMAKE_LINKER "${tools}/bin/armlink.exe")

SET(CMAKE_C_LINK_EXECUTABLE "<CMAKE_LINKER> <LINK_FLAGS> -o <TARGET> <LINK_LIBRARIES> <OBJECTS>")
SET(CMAKE_CXX_LINK_EXECUTABLE "<CMAKE_LINKER> <LINK_FLAGS> -o <TARGET> <LINK_LIBRARIES> <OBJECTS>")

SET(CMAKE_C_RESPONSE_FILE_LINK_FLAG "--via ")
SET(CMAKE_C_OUTPUT_EXTENSION .o)
SET(CMAKE_CXX_OUTPUT_EXTENSION .o)
SET(CMAKE_ASM_OUTPUT_EXTENSION .o)
# When library defined as STATIC, this line is needed to describe how the .a file must be
# create. Some changes to the line may be needed.
SET(CMAKE_C_CREATE_STATIC_LIBRARY "<CMAKE_AR> -r -s --create <TARGET> <LINK_FLAGS> <OBJECTS>" )
SET(CMAKE_CXX_CREATE_STATIC_LIBRARY "<CMAKE_AR> -r -s --create <TARGET> <LINK_FLAGS> <OBJECTS>" )

#SET(CMAKE_ASM_CREATE_STATIC_LIBRARY "${tools}/bin/armar.exe -r -s --create <TARGET> <LINK_FLAGS> <OBJECTS>" )


set(ARMAC6 ON)
# default core

if(NOT ARM_CPU)
    set(
            ARM_CPU "cortex-a5"
            CACHE STRING "Set ARM CPU. Default : cortex-a5"
    )
endif(NOT ARM_CPU)

SET(CMAKE_C_FLAGS "-mcpu=${ARM_CPU} --target=arm-arm-none-eabi" CACHE INTERNAL "C compiler common flags")
SET(CMAKE_CXX_FLAGS "-mcpu=${ARM_CPU} --target=arm-arm-none-eabi" CACHE INTERNAL "C compiler common flags")
SET(CMAKE_ASM_FLAGS "-g -x assembler-with-cpp -masm=auto -mcpu=${ARM_CPU} --target=arm-arm-none-eabi" CACHE INTERNAL "ASM compiler common flags")
#SET(CMAKE_ASM_FLAGS "--cpu=${ARM_CPU}" CACHE INTERNAL "ASM compiler common flags")
#SET(CMAKE_EXE_LINKER_FLAGS "-flto" CACHE INTERNAL "linker flags")

# Where is the target environment
#SET(CMAKE_FIND_ROOT_PATH "${tools}")
# Search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# For libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

#include(CMakePrintHelpers)
#cmake_print_variables(${ASM_DIALECT})
