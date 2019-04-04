#-------------------------------------------------------------------------------
#
# ASIO CONFIDENTIAL
#
# CMakeLists.txt
#
# Build cmsis files for the C-SDK
#
# All contents are strictly proprietary, and not for copying, resale,
# or use outside of the agreed licence.
#
# Copyright © 2011-2018, Asio Ltd.
# All rights reserved.
#
#-------------------------------------------------------------------------------

# Object name in the cmake project
set(OBJECT_NAME cmsis)

# Object file name in the file manager
set(OBJECT_FILE_NAME _${OBJECT_NAME}_all_${VARIANT})

# Add the default source files without taking in account the build type or target
set(SRC
        CMSIS/DSP/Source/BasicMathFunctions/arm_add_f32.c
        CMSIS/DSP/Source/ComplexMathFunctions/arm_cmplx_mag_squared_f32.c
        CMSIS/DSP/Source/FastMathFunctions/arm_cos_f32.c
        CMSIS/DSP/Source/SupportFunctions/arm_fill_f32.c
        CMSIS/DSP/Source/StatisticsFunctions/arm_min_f32.c
        CMSIS/DSP/Source/BasicMathFunctions/arm_mult_f32.c
        CMSIS/DSP/Source/BasicMathFunctions/arm_sub_f32.c
        CMSIS/DSP/Source/TransformFunctions/arm_rfft_fast_f32.c
        CMSIS/DSP/Source/TransformFunctions/arm_rfft_fast_init_f32.c
        CMSIS/DSP/Source/TransformFunctions/arm_cfft_f32.c
        CMSIS/DSP/Source/CommonTables/arm_common_tables.c
        CMSIS/DSP/Source/TransformFunctions/arm_cfft_radix8_f32.c
        )

if (${ARCH} MATCHES armv6m*)
    set(SRC ${SRC}
            CMSIS/DSP/Source/TransformFunctions/arm_bitreversal2.c
            )
elseif(${ARCH} MATCHES armv7m*)
    set(SRC ${SRC}
            CMSIS/DSP/Source/TransformFunctions/arm_bitreversal2.S
            )
endif()

# Create the object gathering
add_library(${OBJECT_FILE_NAME} OBJECT ${SRC})

# Include directories are set in the toolchain files
