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
# Copyright © 2011-2019, Asio Ltd.
# All rights reserved.
#
#-------------------------------------------------------------------------------

# Object name in the cmake project
set(OBJECT_NAME cmsis)

# Object file name in the file manager
set(OBJECT_FILE_NAME _${OBJECT_NAME}_all_${VARIANT})

# Source list for CMSIS
set(CHIRP_CMSIS_SRC)

# Include list for CMSIS
set(CHIRP_CMSIS_INCLUDE)

add_compile_definitions(
        # Try to optimise computation speed
        ARM_MATH_LOOPUNROLL

        # Will allow to only compile some tables (FFT or interpolations ones)
        ARM_DSP_CONFIG_TABLES
        ARM_FAST_ALLOW_TABLES
        ARM_FFT_ALLOW_TABLES

        # For fast cos and sin
        ARM_TABLE_SIN_F32
)

if(IS_FIXED_POINT)
    add_compile_definitions(
            # Only enable FFT tables for the following sizes
            PUBLIC ARM_TABLE_REALCOEF_Q15
            ARM_TABLE_TWIDDLECOEF_Q15_256
            ARM_TABLE_BITREVIDX_FXT_256

            PUBLIC ARM_TABLE_REALCOEF_Q15
            ARM_TABLE_TWIDDLECOEF_Q15_512
            ARM_TABLE_BITREVIDX_FXT_512

            PUBLIC ARM_TABLE_REALCOEF_Q15
            ARM_TABLE_TWIDDLECOEF_Q15_1024
            ARM_TABLE_BITREVIDX_FXT_1024
    )
else()
    add_compile_definitions(
            # Only enable FFT tables for the following sizes
            ARM_TABLE_TWIDDLECOEF_F32_128
            ARM_TABLE_BITREVIDX_FLT_128
            ARM_TABLE_TWIDDLECOEF_RFFT_F32_256
            ARM_TABLE_TWIDDLECOEF_F32_128

            ARM_TABLE_TWIDDLECOEF_F32_256
            ARM_TABLE_BITREVIDX_FLT_256
            ARM_TABLE_TWIDDLECOEF_RFFT_F32_512
            ARM_TABLE_TWIDDLECOEF_F32_256

            ARM_TABLE_TWIDDLECOEF_F32_512
            ARM_TABLE_BITREVIDX_FLT_512
            ARM_TABLE_TWIDDLECOEF_RFFT_F32_1024
            ARM_TABLE_TWIDDLECOEF_F32_512
    )
endif()

# Add the default source files without taking in account the build type or target
set(CHIRP_CMSIS_SRC ${CHIRP_CMSIS_SRC}
        CMSIS/DSP/Source/BasicMathFunctions/arm_add_f32.c
        CMSIS/DSP/Source/FastMathFunctions/arm_cos_f32.c
        CMSIS/DSP/Source/FastMathFunctions/arm_sin_f32.c
        CMSIS/DSP/Source/SupportFunctions/arm_fill_f32.c
        CMSIS/DSP/Source/StatisticsFunctions/arm_min_f32.c
        CMSIS/DSP/Source/BasicMathFunctions/arm_mult_f32.c
        CMSIS/DSP/Source/BasicMathFunctions/arm_sub_f32.c
        CMSIS/DSP/Source/CommonTables/arm_common_tables.c
        CMSIS/DSP/Source/SupportFunctions/arm_q15_to_float.c
        CMSIS/DSP/Source/SupportFunctions/arm_float_to_q15.c
        )

if(IS_FIXED_POINT)
    set(CHIRP_CMSIS_SRC ${CHIRP_CMSIS_SRC}
            CMSIS/DSP/Source/BasicMathFunctions/arm_mult_q15.c
            CMSIS/DSP/Source/BasicMathFunctions/arm_shift_q15.c
            CMSIS/DSP/Source/TransformFunctions/arm_rfft_q15.c
            CMSIS/DSP/Source/TransformFunctions/arm_rfft_init_q15.c
            CMSIS/DSP/Source/TransformFunctions/arm_cfft_q15.c
            CMSIS/DSP/Source/TransformFunctions/arm_cfft_radix4_q15.c
            CMSIS/DSP/Source/TransformFunctions/arm_bitreversal.c
            CMSIS/DSP/Source/CommonTables/arm_const_structs.c
            CMSIS/DSP/Source/ComplexMathFunctions/arm_cmplx_mag_squared_q15.c
            )
else()
    set(CHIRP_CMSIS_SRC ${CHIRP_CMSIS_SRC}
            CMSIS/DSP/Source/TransformFunctions/arm_rfft_fast_f32.c
            CMSIS/DSP/Source/TransformFunctions/arm_rfft_fast_init_f32.c
            CMSIS/DSP/Source/TransformFunctions/arm_cfft_f32.c
            CMSIS/DSP/Source/TransformFunctions/arm_cfft_radix8_f32.c
            CMSIS/DSP/Source/ComplexMathFunctions/arm_cmplx_mag_squared_f32.c
            )
endif()

if (${ARCH} MATCHES armv6m*)
    set(CHIRP_CMSIS_SRC ${CHIRP_CMSIS_SRC}
            CMSIS/DSP/Source/TransformFunctions/arm_bitreversal2.c
            )
elseif(${ARCH} MATCHES armv7m*)
    set(CHIRP_CMSIS_SRC ${CHIRP_CMSIS_SRC}
            CMSIS/DSP/Source/TransformFunctions/arm_bitreversal2.S
            )
endif()

# Convert the relative path to an absolute one - for the include propagation
set(INCLUDES
        CMSIS/Core/Include
        CMSIS/DSP/Include
        )

foreach(INC ${INCLUDES})
    get_filename_component(INC ${INC} ABSOLUTE)
    set(CHIRP_CMSIS_INCLUDE ${CHIRP_CMSIS_INCLUDE} ${INC})
endforeach()

# Create the object gathering
add_library(${OBJECT_FILE_NAME} OBJECT ${CHIRP_CMSIS_SRC})

# Propagate object dependencies to the top level
set(CHIRP_MATHS_OBJECT ${CHIRP_MATHS_OBJECT} ${OBJECT_FILE_NAME} PARENT_SCOPE)

target_include_directories(${OBJECT_FILE_NAME} PRIVATE ${CHIRP_CMSIS_INCLUDE})

set(CHIRP_MATHS_INCLUDE ${CHIRP_MATHS_INCLUDE} ${CHIRP_CMSIS_INCLUDE} PARENT_SCOPE)
